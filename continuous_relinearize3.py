"""continuous_relinearize3.py
------------------------------
2차 수정: 재선형화 구간에서는 그 시점의 "실제 전체 ctrl"을 trim으로
삼아 모든 액추에이터에 균일하게 적용(u = ctrl_trim - K@x) - 특정
관절(hip_abduct_L)에만 따로 ff를 더해서 trim을 이중으로 세는 실수를
없앤다. ff(목표 쪽으로 미는 것)는 램프+사전 정착 구간에서만 쓰고,
재선형화가 시작되면 그 시점에 이미 반영된 값이 trim에 들어있으므로
따로 더하지 않는다.
"""
import math

import numpy as np
import mujoco

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

    TARGET_DEG = -20.0  # 90초 완전 안정 확인된 값 (README 참고)
    RAMP_T = 5.0
    PRE_SETTLE_T = 5.0
    RELIN_INTERVAL = 1.0
    TOTAL_T = 90.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    qpos_eq = qpos_eq0
    qvel_eq = qvel_eq0
    K = K0
    ctrl_trim = np.zeros(nu)
    relin_started = False

    for tick in range(int(TOTAL_T / 0.01)):
        s = min(1.0, data.time / RAMP_T)
        ff = math.radians(TARGET_DEG) * s

        if data.time > RAMP_T + PRE_SETTLE_T:
            if not relin_started:
                relin_started = True
                print(f"t={data.time:.2f}: 주기적 재선형화 시작")
            if tick % int(RELIN_INTERVAL / 0.01) == 0:
                qpos_eq = data.qpos.copy()
                ctrl_trim = data.ctrl.copy()  # 그 시점 실제 ctrl 전체를 trim으로
                K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
                qvel_eq = np.zeros_like(qvel_eq0)

        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
            dv = data.qvel - qvel_eq
            x = np.concatenate([dq, dv])
            if relin_started:
                u = ctrl_trim - K @ x
            else:
                u = -K @ x
                u[a_abduct_L] += ff
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        if tick % 100 == 0:
            zx, zy = common.contact_zmp(model, data)
            t = common.tilt_deg(data.qpos)
            print(f"t={data.time:.2f} s={s:.2f} zmp_y={zy:.3f} tilt={t:.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f}")
        if common.tilt_deg(data.qpos) > 55:
            print("FELL at t=%.2f" % data.time)
            break
    else:
        print(f"완료 - {TOTAL_T}초 안 넘어짐")


if __name__ == "__main__":
    main()
