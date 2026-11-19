"""verify_demo_final_hold.py
-----------------------------
view_sway_demo.py의 마지막 "계속 서있기" 루프(재선형화 추가 수정판)
가 실제로 오래 버티는지 headless로 60초 검증.
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
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")

    LEAN_DEG = 20.0
    RAMP_T = 5.0
    PRE_SETTLE_T = 5.0
    HOLD_T = 10.0
    RELIN_INTERVAL = 1.0
    targets = [-LEAN_DEG, 0.0]

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    K = K0
    qpos_eq = qpos_eq0.copy()
    qvel_eq = qvel_eq0.copy()
    ctrl_trim = np.zeros(nu)
    prev_bias = 0.0
    fell = None

    def step(K_, qpos_eq_, qvel_eq_, ctrl_trim_, hip_R_cmd=None, hip_L_cmd=None):
        nonlocal fell
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
        if common.tilt_deg(data.qpos) > 55 and fell is None:
            fell = data.time

    for cycle in range(len(targets)):
        target_bias = math.radians(targets[cycle])
        n_ramp = int(RAMP_T / 0.01)
        for tick in range(n_ramp):
            bias = prev_bias + (target_bias - prev_bias) * (tick / n_ramp)
            step(K, qpos_eq, qvel_eq, ctrl_trim, min(bias, 0.0), max(bias, 0.0))
            if fell:
                break
        if fell:
            break
        n_settle = int(PRE_SETTLE_T / 0.01)
        for tick in range(n_settle):
            step(K, qpos_eq, qvel_eq, ctrl_trim, min(target_bias, 0.0), max(target_bias, 0.0))
            if fell:
                break
        if fell:
            break
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
            if tick % 20 == 0:
                zx, zy = common.contact_zmp(model, data)
                print(f"    hold t={data.time:.2f} zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f}")
            if fell:
                break
        if fell:
            break
        prev_bias = ctrl_trim[a_abduct_R] if ctrl_trim[a_abduct_R] < 0 else ctrl_trim[a_abduct_L]

    if fell:
        print("2사이클 중 FELL at t=%.2f" % fell)
        return

    print(f"2사이클 통과(t={data.time:.2f}), 이제 '계속 서있기' 60초 검증 시작")
    n_final = int(60.0 / 0.01)
    for tick in range(n_final):
        if tick % int(RELIN_INTERVAL / 0.01) == 0:
            qpos_eq = data.qpos.copy()
            qvel_eq = np.zeros_like(qvel_eq0)
            ctrl_trim = data.ctrl.copy()
            K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
        step(K, qpos_eq, qvel_eq, ctrl_trim)
        if tick % 500 == 0:
            zx, zy = common.contact_zmp(model, data)
            print(f"  final_hold t={data.time:.2f} zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f}")
        if fell:
            break

    if fell:
        print("FELL at t=%.2f" % fell)
    else:
        print("최종 서있기 60초 전부 통과")


if __name__ == "__main__":
    main()
