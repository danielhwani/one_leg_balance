"""continuous_relinearize2.py
------------------------------
1차 시도 수정판: 램프(아직 안 멈춘 전이 상태) 도중에는 재선형화하지
않는다 - 그때의 속도를 "유지할 속도"로 착각하면 안 된다. 램프가
끝나고 원래 K0로 충분히 가라앉힌 다음부터만 주기적 재선형화 시작
(기준 속도는 항상 0으로 강제 - 미세한 잔류 속도를 "의도"로 취급하지
않음).
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

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")

    TARGET_DEG = -20.0
    RAMP_T = 5.0
    PRE_SETTLE_T = 5.0   # 램프 끝나고 K0로 먼저 가라앉히는 시간
    RELIN_INTERVAL = 1.0
    TOTAL_T = 60.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    qpos_eq = qpos_eq0
    qvel_eq = qvel_eq0
    K = K0
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
                ctrl_now = data.ctrl.copy()
                K = relinearize_at(model, qpos_eq, ctrl_now, Q, R)
                qvel_eq = np.zeros_like(qvel_eq0)

        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
            dv = data.qvel - qvel_eq
            x = np.concatenate([dq, dv])
            u = -K @ x
            u[a_abduct_R] += ff
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
