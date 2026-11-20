"""return_toward_center.py
---------------------------
사용자 관찰: "중앙으로 세게 밀면서 재선형화를 반복"하면 실제로
더 중앙에 가깝게 가지만, 너무 오래 하면 불안정해졌다(어제 찾은
"버그"). 그 아이디어를 짧게만 쓰고 멈추면 "더 중앙 + 안정"을 둘 다
얻을 수 있는지 확인한다.

구조: 오른쪽 기울기(ramp+presettle+hold, 기존과 동일) ->
복귀는 "짧은 구간(N초)만 hip_abduct_L을 0으로 강하게 밀면서 매
1초 재선형화" -> 그 다음 override를 끄고 순수 trim으로 길게(60초)
버텨서 진짜 안정적인지 확인. N을 2, 3, 5, 7초로 바꿔가며 어디까지
괜찮은지 스윕.
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


def run(push_duration, verbose=False, FINAL_HOLD_T=60.0):
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")

    LEAN_DEG = 20.0
    RAMP_T = 5.0
    PRE_SETTLE_T = 5.0
    HOLD_T = 10.0
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

    # 1) 오른쪽으로 기울기(기존과 동일).
    n_ramp = int(RAMP_T / 0.01)
    for tick in range(n_ramp):
        bias = -math.radians(LEAN_DEG) * (tick / n_ramp)
        step(K, qpos_eq, qvel_eq, ctrl_trim, bias, None)
        if fell:
            return fell, None
    n_settle = int(PRE_SETTLE_T / 0.01)
    for tick in range(n_settle):
        step(K, qpos_eq, qvel_eq, ctrl_trim, -math.radians(LEAN_DEG), None)
        if fell:
            return fell, None

    # 2) 유지(재선형화).
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
            return fell, None

    # 3) 짧게만 중앙으로 세게 밀기(매 1초 재선형화하며 override=0).
    n_push = int(push_duration / 0.01)
    for tick in range(n_push):
        if tick % int(RELIN_INTERVAL / 0.01) == 0:
            qpos_eq = data.qpos.copy()
            ctrl_trim = data.ctrl.copy()
            K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
        step(K, qpos_eq, qvel_eq, ctrl_trim, 0.0, 0.0)
        if verbose and tick % 20 == 0:
            zx, zy = common.contact_zmp(model, data)
            print(f"    push t={data.time:.2f} zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f}")
        if fell:
            return fell, None

    zx, zy = common.contact_zmp(model, data)
    push_end_tilt = common.tilt_deg(data.qpos)
    push_end_zmp = zy

    # 4) override 끄고 순수 trim + 재선형화로 60초 버티기.
    qpos_eq = data.qpos.copy()
    qvel_eq = np.zeros_like(qvel_eq0)
    ctrl_trim = data.ctrl.copy()
    K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
    n_final = int(FINAL_HOLD_T / 0.01)
    for tick in range(n_final):
        if tick % int(RELIN_INTERVAL / 0.01) == 0:
            qpos_eq = data.qpos.copy()
            ctrl_trim = data.ctrl.copy()
            K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
        step(K, qpos_eq, qvel_eq, ctrl_trim)
        if fell:
            return fell, (push_end_zmp, push_end_tilt)

    zx, zy = common.contact_zmp(model, data)
    return None, (push_end_zmp, push_end_tilt, zy, common.tilt_deg(data.qpos))


if __name__ == "__main__":
    fell, info = run(3.0, verbose=True, FINAL_HOLD_T=120.0)
    print(f"push_duration=3.0s, final_hold=120s: fell={fell} info={info}")
