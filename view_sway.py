"""view_sway.py
---------------
발을 떼기 전에 먼저 좌우로 번갈아 체중을 옮기며 흔드는 모습을
MuJoCo 뷰어로 보여준다. hip_abduct_L(음수 -> 오른쪽으로 체중이동)과
hip_abduct_R(양수 -> 왼쪽으로 체중이동)을 교대로 써서 "오른쪽 기울기
-> 유지 -> 가운데 -> 왼쪽 기울기 -> 유지 -> 가운데 -> ..." 순환.

각 "유지" 구간에서만 continuous_relinearize3.py의 검증된 메커니즘
(1초마다 재선형화)을 쓴다 - 램프(전환) 중에는 재선형화하면 안
된다는 게 오늘의 핵심 교훈.

실행:
  python view_sway.py
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

    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")

    LEAN_DEG = 20.0  # 오늘 90초 검증된 크기
    RAMP_T = 5.0
    HOLD_T = 10.0

    # 순환: 0(가운데) -> -20(오른쪽) -> 0 -> +20(왼쪽) -> 0 -> 반복
    # weight_bias: 음수=오른쪽 쏠림(hip_abduct_L에 들어감), 양수=왼쪽
    # 쏠림(hip_abduct_R에 들어감).
    targets = [-LEAN_DEG, 0.0, LEAN_DEG, 0.0]

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    print("좌우로 번갈아 체중 이동 중... (창을 닫으면 종료)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        prev_bias = 0.0
        cycle = 0
        while viewer.is_running():
            target_bias = math.radians(targets[cycle % len(targets)])
            seg_t0 = data.time

            # --- 전환(램프) 구간: 재선형화 없이 K0 기반으로 부드럽게 ---
            while viewer.is_running() and (data.time - seg_t0) < RAMP_T:
                s = (data.time - seg_t0) / RAMP_T
                bias = prev_bias + (target_bias - prev_bias) * s
                hip_L_cmd = min(bias, 0.0)
                hip_R_cmd = max(bias, 0.0)
                for _ in range(2):
                    dq = np.zeros(nv)
                    mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                    dv = data.qvel - qvel_eq0
                    x = np.concatenate([dq, dv])
                    u = -K0 @ x
                    u[a_abduct_L] += hip_L_cmd
                    u[a_abduct_R] += hip_R_cmd
                    u = np.clip(u, -ctrl_limits, ctrl_limits)
                    data.ctrl[:] = u
                    mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep - 0.0005))

            prev_bias = target_bias

            # --- 유지 구간: 1초마다 재선형화해서 그 자세를 계속 붙잡기 ---
            qpos_eq = data.qpos.copy()
            qvel_eq = np.zeros_like(qvel_eq0)
            ctrl_trim = data.ctrl.copy()
            K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
            hold_t0 = data.time
            relin_tick = 0
            while viewer.is_running() and (data.time - hold_t0) < HOLD_T:
                if relin_tick % 100 == 0:
                    qpos_eq = data.qpos.copy()
                    ctrl_trim = data.ctrl.copy()
                    K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
                for _ in range(2):
                    dq = np.zeros(nv)
                    mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
                    dv = data.qvel - qvel_eq
                    x = np.concatenate([dq, dv])
                    u = np.clip(ctrl_trim - K @ x, -ctrl_limits, ctrl_limits)
                    data.ctrl[:] = u
                    mujoco.mj_step(model, data)
                    relin_tick += 1
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep - 0.0005))

            cycle += 1


if __name__ == "__main__":
    main()
