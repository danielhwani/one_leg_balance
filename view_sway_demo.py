"""view_sway_demo.py
---------------------
오늘(2026-11-19) 검증된 레시피를 MuJoCo 뷰어로 보여주는 데모.
ROS2 없이 순수 터미널 + MuJoCo 창만 쓴다.

구조:
1. 오른쪽으로 기울기(램프 5초 -> 정착 5초, K0 기반 override) -> 그
   자세를 재선형화하며 10초 유지.
2. 복귀: "짧게(3초)만 override=0으로 세게 밀면서 매 1초 재선형화"
   -> 그 다음부터는 override 없이 순수 trim + 계속 재선형화로 계속
   서있기. (사용자 관찰 - "버그였던 버전이 오히려 중앙에 더
   가까웠다"에서 착안: 짧게 세게 밀고 나서도 재선형화를 계속
   이어가면 더 중앙에(~0.4도) 가깝게, 그리고 안정적으로 선다 -
   원래 버그는 "밀기"는 맞았는데 그 뒤 재선형화를 완전히 멈춘 게
   문제였다.)
"""
import math
import time

import numpy as np
import mujoco
import mujoco.viewer

import common


def relinearize_at(model, qpos, ctrl, Q, R):
    nv, nu = model.nv, model.nu
    data_lin = mujoco.MjData(model)
    data_lin.qpos[:] = qpos
    data_lin.qvel[:] = 0.0
    data_lin.ctrl[:] = ctrl
    mujoco.mj_forward(model, data_lin)
    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data_lin, 1e-6, True, A, B, None, None)
    K = common.solve_dare(A, B, Q, R)
    return K


def main():
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")

    LEAN_DEG = 20.0
    RAMP_T = 5.0
    PRE_SETTLE_T = 5.0
    HOLD_T = 10.0
    PUSH_BACK_T = 3.0
    RELIN_INTERVAL = 1.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    K = K0
    qpos_eq = qpos_eq0.copy()
    qvel_eq = qvel_eq0.copy()
    ctrl_trim = np.zeros(nu)

    print("오른쪽으로 기울기 -> 유지 -> 중앙 근처로 복귀 데모. 창을 닫으면 종료.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        def step(K_, qpos_eq_, qvel_eq_, ctrl_trim_, hip_R_cmd=None, hip_L_cmd=None):
            # hip_R_cmd/hip_L_cmd가 None이면 override 없이 순수
            # trim만 쓴다 - (0.0, 0.0)을 넘기면 "기울기를 강제로
            # 0으로 지우는" override가 된다(원래는 return 단계에서만
            # 의도적으로 씀).
            ctrl_trim_eff = ctrl_trim_.copy()
            if hip_R_cmd is not None:
                ctrl_trim_eff[a_abduct_R] = hip_R_cmd
            if hip_L_cmd is not None:
                ctrl_trim_eff[a_abduct_L] = hip_L_cmd
            for _ in range(2):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq_, data.qpos)
                dv = data.qvel - qvel_eq_
                x = np.concatenate([dq, dv])
                u = np.clip(ctrl_trim_eff - K_ @ x, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - 0.0005))

        # === 1) 오른쪽으로 기울기 ===
        n_ramp = int(RAMP_T / 0.01)
        for tick in range(n_ramp):
            if not viewer.is_running():
                return
            bias = -math.radians(LEAN_DEG) * (tick / n_ramp)
            step(K, qpos_eq, qvel_eq, ctrl_trim, bias, None)

        n_settle = int(PRE_SETTLE_T / 0.01)
        for tick in range(n_settle):
            if not viewer.is_running():
                return
            step(K, qpos_eq, qvel_eq, ctrl_trim, -math.radians(LEAN_DEG), None)

        # === 2) 유지 (재선형화) ===
        qpos_eq = data.qpos.copy()
        qvel_eq = np.zeros_like(qvel_eq0)
        ctrl_trim = data.ctrl.copy()
        K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
        n_hold = int(HOLD_T / 0.01)
        for tick in range(n_hold):
            if not viewer.is_running():
                return
            if tick % int(RELIN_INTERVAL / 0.01) == 0:
                qpos_eq = data.qpos.copy()
                ctrl_trim = data.ctrl.copy()
                K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
            step(K, qpos_eq, qvel_eq, ctrl_trim)

        print("복귀 시작 - 짧게 밀면서 계속 재선형화...")

        # === 3) 복귀: 짧게(3초)만 override=0으로 밀면서 매 1초 재선형화 ===
        n_push = int(PUSH_BACK_T / 0.01)
        for tick in range(n_push):
            if not viewer.is_running():
                return
            if tick % int(RELIN_INTERVAL / 0.01) == 0:
                qpos_eq = data.qpos.copy()
                ctrl_trim = data.ctrl.copy()
                K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
            step(K, qpos_eq, qvel_eq, ctrl_trim, 0.0, 0.0)

        print("데모 끝 - 계속 서있는 모습을 보실 수 있습니다 (창을 닫으면 종료).")

        # === 4) override 끄고 순수 trim + 계속 재선형화로 계속 서있기 ===
        qpos_eq = data.qpos.copy()
        qvel_eq = np.zeros_like(qvel_eq0)
        ctrl_trim = data.ctrl.copy()
        K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
        hold_tick = 0
        while viewer.is_running():
            if hold_tick % int(RELIN_INTERVAL / 0.01) == 0:
                qpos_eq = data.qpos.copy()
                qvel_eq = np.zeros_like(qvel_eq0)
                ctrl_trim = data.ctrl.copy()
                K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
            hold_tick += 1
            step(K, qpos_eq, qvel_eq, ctrl_trim)


if __name__ == "__main__":
    main()
