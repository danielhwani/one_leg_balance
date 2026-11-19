"""view_sway_demo.py
---------------------
오늘(2026-11-19) 검증된 범위 - 오른쪽으로 기울어서 버티다가 중앙
근처로 돌아오는 것까지 - 를 MuJoCo 뷰어로 보여주는 데모. ROS2 없이
순수 터미널 + MuJoCo 창만 쓴다.

verify_sway2.py에서 검증된 정확한 레시피(램프 -> 5초 정착 -> 1초마다
재선형화하며 유지 -> 다음 전환은 "실제 달성값"에서 시작)를 그대로
가져왔다. 3번째 전환(반대쪽으로 더 가는 것)은 아직 불안정해서
(README의 "다음 단계" 참고) 데모에는 2단계(기울기+복귀)까지만 넣었다.

실행:
  python view_sway_demo.py
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
    RELIN_INTERVAL = 1.0
    targets = [-LEAN_DEG, 0.0]  # 오늘 검증된 범위만

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    K = K0
    qpos_eq = qpos_eq0.copy()
    qvel_eq = qvel_eq0.copy()
    ctrl_trim = np.zeros(nu)
    prev_bias = 0.0

    print("오른쪽으로 기울기 -> 유지 -> 중앙 근처로 복귀 데모. 창을 닫으면 종료.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        def step(K_, qpos_eq_, qvel_eq_, ctrl_trim_, hip_R_cmd=None, hip_L_cmd=None):
            # hip_R_cmd/hip_L_cmd가 None이면(hold 단계) override 없이
            # 순수 trim만 쓴다 - (0.0, 0.0)을 넘기면 "기울기를 강제로
            # 0으로 지우는" 것과 같아서(오늘 찾은 버그, 사용자가 본
            # "거의 중앙으로 돌아와서 유지하다 쓰러짐" 증상의 원인).
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

        for cycle in range(len(targets)):
            if not viewer.is_running():
                break
            target_bias = math.radians(targets[cycle])

            n_ramp = int(RAMP_T / 0.01)
            for tick in range(n_ramp):
                if not viewer.is_running():
                    break
                bias = prev_bias + (target_bias - prev_bias) * (tick / n_ramp)
                step(K, qpos_eq, qvel_eq, ctrl_trim, min(bias, 0.0), max(bias, 0.0))

            n_settle = int(PRE_SETTLE_T / 0.01)
            for tick in range(n_settle):
                if not viewer.is_running():
                    break
                step(K, qpos_eq, qvel_eq, ctrl_trim, min(target_bias, 0.0), max(target_bias, 0.0))

            qpos_eq = data.qpos.copy()
            qvel_eq = np.zeros_like(qvel_eq0)
            ctrl_trim = data.ctrl.copy()
            K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
            n_hold = int(HOLD_T / 0.01)
            for tick in range(n_hold):
                if not viewer.is_running():
                    break
                if tick % int(RELIN_INTERVAL / 0.01) == 0:
                    qpos_eq = data.qpos.copy()
                    ctrl_trim = data.ctrl.copy()
                    K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
                step(K, qpos_eq, qvel_eq, ctrl_trim)

            prev_bias = ctrl_trim[a_abduct_R] if ctrl_trim[a_abduct_R] < 0 else ctrl_trim[a_abduct_L]

        print("데모 끝 - 계속 서있는 모습을 보실 수 있습니다 (창을 닫으면 종료).")
        hold_tick = 0
        while viewer.is_running():
            # 여기서도 1초마다 계속 재선형화해야 한다 - K를 한 번
            # 정한 채로 오래 고정해두면(여기선 "계속") 그 자체가
            # 서서히 발산한다(오늘 15초 램프 테스트에서 겪은 것과
            # 같은 버그 - 데모 마지막 루프에 깜빡하고 안 넣었었다).
            if hold_tick % int(RELIN_INTERVAL / 0.01) == 0:
                qpos_eq = data.qpos.copy()
                qvel_eq = np.zeros_like(qvel_eq0)
                ctrl_trim = data.ctrl.copy()
                K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
            hold_tick += 1
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
            dv = data.qvel - qvel_eq
            x = np.concatenate([dq, dv])
            u = np.clip(ctrl_trim - K @ x, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - 0.0005))


if __name__ == "__main__":
    main()
