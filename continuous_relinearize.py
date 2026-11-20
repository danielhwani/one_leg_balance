"""continuous_relinearize.py
-----------------------------
완벽한 평형점을 찾으려 하지 않고, 주기적으로(RELIN_INTERVAL마다)
"지금 상태"를 새 기준점으로 삼아 재선형화 + 새 K를 계산 - 느린
발산이 누적되기 전에 계속 따라가면서 잡는다(successive
linearization, 일종의 저렴한 비선형 제어).

hip_abduct_L에 피드포워드 bias를 계속 걸어서 체중 이동을 유도하고,
그 위에서 주기적 재선형화가 "안 넘어지게" 잡아주는지 확인한다.
"""
import math

import numpy as np
import mujoco

import common


def relinearize_at(model, qpos, qvel, ctrl, Q, R):
    nv, nu = model.nv, model.nu
    data_lin = mujoco.MjData(model)
    data_lin.qpos[:] = qpos
    data_lin.qvel[:] = qvel
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
    Q, R = common.build_cost_two_leg(model)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")

    TARGET_DEG = -20.0
    RAMP_T = 5.0
    RELIN_INTERVAL = 1.0  # 초

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    qpos_eq = qpos_eq0.copy()
    qvel_eq = qvel_eq0.copy()
    ctrl_trim = np.zeros(nu)
    K = relinearize_at(model, qpos_eq, qvel_eq, ctrl_trim, Q, R)

    TOTAL_T = 60.0
    for tick in range(int(TOTAL_T / 0.01)):
        s = min(1.0, data.time / RAMP_T)
        ff = math.radians(TARGET_DEG) * s

        # 주기적 재선형화 - "지금 상태"를 새 기준으로.
        if tick > 0 and tick % int(RELIN_INTERVAL / 0.01) == 0:
            qpos_eq = data.qpos.copy()
            qvel_eq = data.qvel.copy()
            ctrl_trim = data.ctrl.copy()
            K = relinearize_at(model, qpos_eq, qvel_eq, ctrl_trim, Q, R)

        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
            dv = data.qvel - qvel_eq
            x = np.concatenate([dq, dv])
            # hip_abduct_L은 계속 램프되는 목표(ff)로 직접 몰고(열린 루프),
            # 나머지는 재선형화된 K로 "지금 기준점 대비 오차"를 교정.
            u = -K @ x
            u[a_abduct_R] = ff
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
