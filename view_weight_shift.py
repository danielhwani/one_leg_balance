"""view_weight_shift.py
------------------------
continuous_relinearize3.py에서 검증된(90초 완전 안정) 체중이동
메커니즘을 MuJoCo 자체 뷰어로 직접 보는 스크립트. ROS2/RViz2와
무관한 순수 로컬 창 - computer_graphics 프로젝트의
view_humanoid_stand.py와 같은 이유(디스커버리 문제 없이 바로 확인).

실행:
  python view_weight_shift.py

창을 닫으면 종료. 오른다리(hip_abduct_R)가 천천히 기울며 체중을
오른쪽으로 옮기고, 그 뒤로는 1초마다 "지금 상태"를 새 기준점으로
재선형화하면서 그 자세를 계속 유지한다(README의 교훈 참고).
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

    TARGET_DEG = -20.0
    RAMP_T = 5.0
    PRE_SETTLE_T = 5.0
    RELIN_INTERVAL = 1.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    qpos_eq = qpos_eq0
    qvel_eq = qvel_eq0
    K = K0
    ctrl_trim = np.zeros(nu)
    relin_started = False

    print("오른다리로 체중 이동 중... (창을 닫으면 종료)")
    tick = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            s = min(1.0, data.time / RAMP_T)
            ff = math.radians(TARGET_DEG) * s

            if data.time > RAMP_T + PRE_SETTLE_T:
                if not relin_started:
                    relin_started = True
                    print(f"t={data.time:.2f}: 재선형화 시작 - 이 자세를 계속 유지")
                if tick % int(RELIN_INTERVAL / 0.01) == 0:
                    qpos_eq = data.qpos.copy()
                    ctrl_trim = data.ctrl.copy()
                    K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
                    qvel_eq = np.zeros_like(qvel_eq0)

            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
            dv = data.qvel - qvel_eq
            x = np.concatenate([dq, dv])
            if relin_started:
                u = ctrl_trim - K @ x
            else:
                u = -K @ x
                u[a_abduct_R] += ff
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u

            mujoco.mj_step(model, data)
            viewer.sync()
            tick += 1
            time.sleep(max(0.0, model.opt.timestep - 0.0005))


if __name__ == "__main__":
    main()
