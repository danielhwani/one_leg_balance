import math

import numpy as np
import mujoco

import common


def main():
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")

    # 1단계: K0 + 피드포워드로 "거의 정적인 체중이동 상태"까지 이동.
    TARGET_DEG = -20.0
    RAMP_T = 5.0
    SETTLE_T = 40.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    for tick in range(int(SETTLE_T / 0.01)):
        s = min(1.0, data.time / RAMP_T)
        ff = math.radians(TARGET_DEG) * s
        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = -K0 @ x
            u[a_abduct_L] += ff
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        if tick % 200 == 0:
            zx, zy = common.contact_zmp(model, data)
            print(f"  1단계 t={data.time:.2f} zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f} "
                  f"|v|={np.linalg.norm(data.qvel[:3]):.5f}")

    zx, zy = common.contact_zmp(model, data)
    print(f"1단계 끝(t={data.time:.2f}): zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f} "
          f"|v|={np.linalg.norm(data.qvel[:3]):.4f}")

    # 2단계: 지금 도달한 상태를 새 평형점으로 삼아 재선형화 + 새 K.
    # common.linearize()는 ctrl=0에서 선형화하는데, 이 지점은 ctrl=0이
    # 아니라 trim(ff_final)이 "u=0"에 해당하는 점이다 - 그 trim을
    # 적용한 상태에서 선형화해야 한다(이전 시도가 이걸 빼먹어서 실패).
    qpos_eq1 = data.qpos.copy()
    qvel_eq1 = data.qvel.copy()
    ff_final = math.radians(TARGET_DEG)
    data_lin = mujoco.MjData(model)
    data_lin.qpos[:] = qpos_eq1
    data_lin.qvel[:] = qvel_eq1
    data_lin.ctrl[:] = 0.0
    data_lin.ctrl[a_abduct_L] = ff_final
    mujoco.mj_forward(model, data_lin)
    A1 = np.zeros((2 * nv, 2 * nv))
    B1 = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data_lin, 1e-6, True, A1, B1, None, None)
    K1 = common.solve_dare(A1, B1, Q, R)
    print("재선형화 완료(trim 포함), 새 K1 계산됨")

    # 3단계: 이 자세를 "유지"하는 데 필요했던 trim(ff)은 그대로 두고,
    # K1(새 평형점 기준) 교정만 추가 - 평형점에서조차 ctrl=0이 아니라
    # 이 trim이 "u=0"에 해당한다는 걸 빼먹으면(이전 시도) 바로 넘어짐.
    ff_final = math.radians(TARGET_DEG)
    for tick in range(int(30.0 / 0.01)):
        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq1, data.qpos)
            dv = data.qvel - qvel_eq1
            x = np.concatenate([dq, dv])
            u = -K1 @ x
            u[a_abduct_L] += ff_final
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        if tick % 100 == 0:
            zx, zy = common.contact_zmp(model, data)
            t = common.tilt_deg(data.qpos)
            print(f"  K1 유지중 t={data.time:.2f} zmp_y={zy:.3f} tilt={t:.2f}")
        if common.tilt_deg(data.qpos) > 55:
            print("K1 전환 후 FELL at t=%.2f" % data.time)
            break
    else:
        print("K1 전환 후 30초 안정 - 성공")


if __name__ == "__main__":
    main()
