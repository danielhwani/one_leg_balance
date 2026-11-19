"""view_sway_demo.py(양쪽 버전)와 똑같은 로직을 뷰어 없이 검증."""
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
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    qadr_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_R")]
    qadr_L = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]
    act_qadr = {a_abduct_R: qadr_R, a_abduct_L: qadr_L}

    LEAN_DEG, RAMP_T, PRE_SETTLE_T, HOLD_T, STAND_T, RELIN_INTERVAL = 10.0, 5.0, 5.0, 10.0, 60.0, 1.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    fell = None

    def step(K_, qpos_eq_, qvel_eq_, ctrl_trim_, act_id=None, cmd=None):
        nonlocal fell
        ctrl_trim_eff = ctrl_trim_.copy()
        if act_id is not None:
            ctrl_trim_eff[act_id] = cmd
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

    def lean_and_return(act_id, sign, label):
        K, qpos_eq, qvel_eq, ctrl_trim = K0, qpos_eq0, qvel_eq0, np.zeros(nu)
        n_ramp = int(RAMP_T / 0.01)
        for tick in range(n_ramp):
            bias = sign * math.radians(LEAN_DEG) * (tick / n_ramp)
            step(K, qpos_eq, qvel_eq, ctrl_trim, act_id, bias)
            if fell:
                return False
        n_settle = int(PRE_SETTLE_T / 0.01)
        for tick in range(n_settle):
            step(K, qpos_eq, qvel_eq, ctrl_trim, act_id, sign * math.radians(LEAN_DEG))
            if fell:
                return False

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
                return False

        zx, zy = common.contact_zmp(model, data)
        actual_deg = math.degrees(data.qpos[act_qadr[act_id]])
        print(f"{label} 유지 끝: t={data.time:.2f} zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f} "
              f"actual={actual_deg:.2f}deg |v|={np.linalg.norm(data.qvel[:3]):.4f}")

        n_stand = int(STAND_T / 0.01)
        for tick in range(n_stand):
            step(K0, qpos_eq0, qvel_eq0, np.zeros(nu))
            if tick % 500 == 0:
                print(f"    [{label} K0복귀] t={data.time:.2f} tilt={common.tilt_deg(data.qpos):.2f} "
                      f"hip_L={math.degrees(data.qpos[qadr_L]):.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f}")
            if fell:
                return False
        print(f"{label} K0 복귀 끝: t={data.time:.2f} tilt={common.tilt_deg(data.qpos):.2f}")
        return True

    if not lean_and_return(a_abduct_R, -1.0, "오른쪽"):
        print("FELL(right) at", fell); return
    print(f"  (왼쪽 시작 전 상태) hip_R={math.degrees(data.qpos[qadr_R]):.3f}deg "
          f"hip_L={math.degrees(data.qpos[qadr_L]):.3f}deg |v|={np.linalg.norm(data.qvel[:3]):.5f}")
    if not lean_and_return(a_abduct_L, 1.0, "왼쪽"):
        print("FELL(left) at", fell); return

    n_final = int(60.0 / 0.01)
    for tick in range(n_final):
        step(K0, qpos_eq0, qvel_eq0, np.zeros(nu))
        if tick % 1000 == 0:
            print(f"  final t={data.time:.2f} tilt={common.tilt_deg(data.qpos):.3f}")
        if fell:
            print("FELL(final)", fell); return

    print("양쪽(오른쪽->왼쪽) + 60초 최종 검증 통과")


if __name__ == "__main__":
    main()
