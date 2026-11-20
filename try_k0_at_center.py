"""try_k0_at_center.py
-----------------------
return_toward_center.py로 기울기 ~0.4도까지 돌아온 뒤, 거기서부터는
계속 재선형화하는 대신 **원래 검증된 K0(두 발 서있기)로 전환**해서
똑바로 서는지 확인. 지금은 상태가 K0의 평형점(qpos_eq0)에 훨씬
가까워서(예전엔 기울기 4.7도+엉덩이 몇 도 차이였지만 지금은 0.4도
근처) 충격 없이 버틸 가능성이 있다.
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

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    qadr_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]

    LEAN_DEG = 10.0
    RAMP_T = 5.0
    PRE_SETTLE_T = 5.0
    HOLD_T = 10.0
    PUSH_BACK_T = 3.0
    RELIN_INTERVAL = 1.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    fell = None

    def step(K_, qpos_eq_, qvel_eq_, ctrl_trim_, hip_L_cmd=None, hip_R_cmd=None):
        nonlocal fell
        ctrl_trim_eff = ctrl_trim_.copy()
        if hip_L_cmd is not None:
            ctrl_trim_eff[a_abduct_R] = hip_L_cmd
        if hip_R_cmd is not None:
            ctrl_trim_eff[a_abduct_L] = hip_R_cmd
        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq_, data.qpos)
            dv = data.qvel - qvel_eq_
            x = np.concatenate([dq, dv])
            u = np.clip(ctrl_trim_eff - K_ @ x, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        if common.tilt_deg(data.qpos) > 55 and fell is None:
            fell = data.time

    K, qpos_eq, qvel_eq, ctrl_trim = K0, qpos_eq0, qvel_eq0, np.zeros(nu)

    # 1) 오른쪽 기울기 + 정착
    n_ramp = int(RAMP_T / 0.01)
    for tick in range(n_ramp):
        bias = -math.radians(LEAN_DEG) * (tick / n_ramp)
        step(K, qpos_eq, qvel_eq, ctrl_trim, bias, None)
        if fell:
            print("FELL(ramp) at", fell)
            return
    n_settle = int(PRE_SETTLE_T / 0.01)
    for tick in range(n_settle):
        step(K, qpos_eq, qvel_eq, ctrl_trim, -math.radians(LEAN_DEG), None)
        if fell:
            print("FELL(presettle) at", fell)
            return

    zx, zy = common.contact_zmp(model, data)
    print(f"ramp+presettle 끝: t={data.time:.2f} zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f} "
          f"actual_hip_L={math.degrees(data.qpos[qadr_R]):.2f}deg |v|={np.linalg.norm(data.qvel[:3]):.4f}")

    # 2) 유지
    qpos_eq = data.qpos.copy()
    qvel_eq = np.zeros_like(qvel_eq0)
    ctrl_trim = data.ctrl.copy()
    K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
    n_hold = int(HOLD_T / 0.01)
    for tick in range(n_hold):
        if tick % int(RELIN_INTERVAL / 0.01) == 0:
            qpos_eq = data.qpos.copy()
            ctrl_trim = data.ctrl.copy()
            K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
        step(K, qpos_eq, qvel_eq, ctrl_trim)
        if fell:
            print("FELL(hold) at", fell)
            return

    zx, zy = common.contact_zmp(model, data)
    print(f"hold 끝: t={data.time:.2f} zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f} "
          f"actual_hip_L={math.degrees(data.qpos[qadr_R]):.2f}deg |v|={np.linalg.norm(data.qvel[:3]):.4f}")
    # 10도 기울기는 hold 끝난 지점이 이미 K0 범위(3~4도) 안이라
    # "밀기" 단계 자체를 건너뛴다 - 밀기 단계가 오히려 더 멀어지게
    # 만들었다(오늘 찾은 버그: 작은 오프셋에서 밀기가 과도하게
    # 오버슈트함).

    # 4) 사용자 제안: 지금 qpos/qvel만 깨끗하게 가져와서 완전히 새
    #    MjData로 리셋한 뒤 순수 K0를 적용 - 혹시 지금 data 객체에
    #    숨겨진 상태(접촉 warmstart, act 등)가 남아있어서 방해하는
    #    건지 확인.
    print("=== 완전히 새 MjData로 리셋 + 순수 K0 ===")
    clean_qpos = data.qpos.copy()
    clean_qvel = data.qvel.copy()
    data = mujoco.MjData(model)
    data.qpos[:] = clean_qpos
    data.qvel[:] = clean_qvel
    mujoco.mj_forward(model, data)
    print(f"  리셋 직후: tilt={common.tilt_deg(data.qpos):.2f} actual_hip_L={math.degrees(data.qpos[qadr_R]):.2f}deg "
          f"act_norm={np.linalg.norm(data.act):.4f} ncon={data.ncon}")

    n_final = int(120.0 / 0.01)
    for tick in range(n_final):
        step(K0, qpos_eq0, qvel_eq0, np.zeros(nu))
        if tick % 500 == 0:
            zx, zy = common.contact_zmp(model, data)
            print(f"  K0 t={data.time:.2f} zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f} "
                  f"actual_hip_L={math.degrees(data.qpos[qadr_R]):.2f}deg |v|={np.linalg.norm(data.qvel[:3]):.4f}")
        if fell:
            print("FELL(K0) at", fell)
            return

    print("K0 전환 후 120초 전부 통과 - 똑바로 서기 성공!")


if __name__ == "__main__":
    main()
