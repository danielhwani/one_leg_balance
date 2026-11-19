"""verify_demo_v2.py - view_sway_demo.py(10도+K0 버전)와 똑같은 로직을
뷰어 없이 60초 검증."""
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
    return common.solve_dare(A, B, Q, R)


def main():
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)
    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")

    LEAN_DEG, RAMP_T, PRE_SETTLE_T, HOLD_T, RELIN_INTERVAL = 10.0, 5.0, 5.0, 10.0, 1.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    K, qpos_eq, qvel_eq, ctrl_trim = K0, qpos_eq0.copy(), qvel_eq0.copy(), np.zeros(nu)
    fell = None

    def step(K_, qpos_eq_, qvel_eq_, ctrl_trim_, hip_R_cmd=None):
        nonlocal fell
        ctrl_trim_eff = ctrl_trim_.copy()
        if hip_R_cmd is not None:
            ctrl_trim_eff[a_abduct_R] = hip_R_cmd
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

    n_ramp = int(RAMP_T / 0.01)
    for tick in range(n_ramp):
        bias = -math.radians(LEAN_DEG) * (tick / n_ramp)
        step(K, qpos_eq, qvel_eq, ctrl_trim, bias)
        if fell:
            print("FELL(ramp)", fell); return
    n_settle = int(PRE_SETTLE_T / 0.01)
    for tick in range(n_settle):
        step(K, qpos_eq, qvel_eq, ctrl_trim, -math.radians(LEAN_DEG))
        if fell:
            print("FELL(presettle)", fell); return

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
            print("FELL(hold)", fell); return

    zx, zy = common.contact_zmp(model, data)
    print(f"hold 끝: t={data.time:.2f} tilt={common.tilt_deg(data.qpos):.2f}")

    n_final = int(60.0 / 0.01)
    for tick in range(n_final):
        step(K0, qpos_eq0, qvel_eq0, np.zeros(nu))
        if tick % 1000 == 0:
            print(f"  K0 t={data.time:.2f} tilt={common.tilt_deg(data.qpos):.3f}")
        if fell:
            print("FELL(K0)", fell); return

    print("view_sway_demo.py 로직 60초 검증 통과")


if __name__ == "__main__":
    main()
