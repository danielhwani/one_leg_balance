"""tvlqr_lift_ablation.py
--------------------------
tvlqr_lift_v2_widened.py(폭0.40+발오프셋+kp/kv x3 전부)가 좁은 원래
설정(2.61s)보다 훨씬 나쁘다(1.56s, 심지어 t=0.5~0.7s에 거의 쓰러지기도
전에 fz가 요동친다) - 어느 변경이 범인인지 분리한다:
  B: 폭0.40+발오프셋, kp/kv 기본값(x1) - 폭/발만의 효과
  D: 좁은 폭(원래), kp/kv x3 - 다리 강성만의 효과
평형점은 기하학(관절각)에만 좌우되므로 kp/kv와 무관 - 폭에 맞는
평형점 파일만 그대로 재사용한다(widen: best_one_leg_qpos_current_gn.npy,
narrow: best_one_leg_qpos_narrow.npy는 다른 스윙포즈라 narrow_gn 필요).
"""
import math

import numpy as np
import mujoco

import common
from pelvis_width_test import widen_pelvis
from foot_offset_test import set_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv
from find_one_leg_equilibrium_v2 import mat2quat
from find_one_leg_eq_current import solve_torso_for_lean_general
from tvlqr_lift_v1 import smoothstep, linearize_at, foot_fz


def run_variant(tag, widen, feet, kp_kv_scale, eq_path, T_transition=4.0, hold_s=5.0,
                 verbose=False, LEAN_FRAC=0.35, SWING_TRACK_WEIGHT=30.0):
    model = common.load_model()
    if widen:
        widen_pelvis(model, 0.40)
    if feet:
        set_feet(model, 0.28, 0.16, 0.0)
    if kp_kv_scale != 1.0:
        scale_kp_kv(model, LEG_ACTUATORS, kp_kv_scale)
    model.opt.timestep = 0.0005
    nv, nu = model.nv, model.nu
    n_sub = round(0.01 / model.opt.timestep)
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    qpos_eq1 = np.load(eq_path)

    data0 = mujoco.MjData(model)
    data0.qpos[:] = qpos_eq0
    mujoco.mj_forward(model, data0)
    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    target_foot_pos = data0.xpos[foot_L_id].copy()
    target_foot_mat = data0.xmat[foot_L_id].copy().reshape(3, 3)
    leg_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_L")
    pelvis_offset = model.body_pos[leg_L_id].copy()

    def jq(name):
        return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]

    q_abduct_L = jq("hip_abduct_L")
    q_hip_L, q_knee_L = jq("hip_L"), jq("knee_L")
    q_ankle_roll_L, q_ankle_pitch_L = jq("ankle_roll_L"), jq("ankle_pitch_L")
    q_abduct_R, q_hip_R, q_knee_R = jq("hip_abduct_R"), jq("hip_R"), jq("knee_R")
    q_waist_roll, q_waist_pitch = jq("waist_roll"), jq("waist_pitch")

    theta2 = qpos_eq0[q_hip_L]
    theta3 = qpos_eq0[q_knee_L]
    theta4 = qpos_eq0[q_ankle_roll_L]
    theta5 = qpos_eq0[q_ankle_pitch_L]
    theta1_final = qpos_eq1[q_abduct_L]
    waist_roll_0, waist_roll_final = qpos_eq0[q_waist_roll], qpos_eq1[q_waist_roll]
    waist_pitch_0, waist_pitch_final = qpos_eq0[q_waist_pitch], qpos_eq1[q_waist_pitch]
    abduct_R_final = qpos_eq1[q_abduct_R]
    hip_R_final = qpos_eq1[q_hip_R]
    knee_R_final = qpos_eq1[q_knee_R]

    def build_qpos_at(s_global):
        lean_s = smoothstep(min(1.0, s_global / LEAN_FRAC))
        swing_s = smoothstep(s_global)
        theta1 = theta1_final * lean_s
        P0, R0 = solve_torso_for_lean_general(theta1, theta2, theta3, theta4, theta5,
                                               target_foot_pos, target_foot_mat, pelvis_offset)
        quat0 = mat2quat(R0)
        qpos = qpos_eq0.copy()
        qpos[0:3] = P0
        qpos[3:7] = quat0
        qpos[q_abduct_L] = theta1
        qpos[q_waist_roll] = waist_roll_0 + (waist_roll_final - waist_roll_0) * lean_s
        qpos[q_waist_pitch] = waist_pitch_0 + (waist_pitch_final - waist_pitch_0) * lean_s
        qpos[q_abduct_R] = abduct_R_final * swing_s
        qpos[q_hip_R] = hip_R_final * swing_s
        qpos[q_knee_R] = knee_R_final * swing_s
        return qpos

    def ctrl_of(qpos):
        c = np.zeros(nu)
        for i in range(nu):
            jid = model.actuator_trnid[i, 0]
            qadr = model.jnt_qposadr[jid]
            c[i] = qpos[qadr]
        return c

    Q, R = common.build_cost_two_leg(model)
    ctrl_limits = np.array([math.radians(30.0)] * nu)
    for jname in ["hip_abduct_R", "hip_R", "knee_R"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dofadr = model.jnt_dofadr[jid]
        Q[dofadr, dofadr] *= SWING_TRACK_WEIGHT
        Q[nv + dofadr, nv + dofadr] *= SWING_TRACK_WEIGHT

    n_wp = 40
    wps = [build_qpos_at(i / n_wp) for i in range(n_wp + 1)]
    ctrl_wps = [ctrl_of(q) for q in wps]
    A_list, B_list = [], []
    for q, c in zip(wps, ctrl_wps):
        A, B = linearize_at(model, q, c, nv, nu)
        A_list.append(A)
        B_list.append(B)
    P = Q.copy()
    K_list = [None] * (n_wp + 1)
    A_end, B_end = A_list[-1], B_list[-1]
    for _ in range(200):
        BtP = B_end.T @ P
        K_end = np.linalg.solve(R + BtP @ B_end, BtP @ A_end)
        P = Q + A_end.T @ P @ A_end - A_end.T @ P @ B_end @ K_end
    K_list[n_wp] = K_end
    for k in range(n_wp - 1, -1, -1):
        A, B = A_list[k], B_list[k]
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        K_list[k] = K
    K1 = K_list[n_wp]
    dt_wp = T_transition / n_wp

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    liftoff_t = None
    fell_t = None
    for tick in range(int((T_transition + hold_s) / 0.01)):
        t = data.time
        if t < T_transition:
            k = min(n_wp - 1, int(t / dt_wp))
            s_local = (t - k * dt_wp) / dt_wp
            qpos_ref = (1 - s_local) * wps[k] + s_local * wps[k + 1]
            qpos_ref[3:7] = wps[k][3:7] * (1 - s_local) + wps[k + 1][3:7] * s_local
            n = np.linalg.norm(qpos_ref[3:7])
            if n > 1e-9:
                qpos_ref[3:7] /= n
            ctrl_ref = (1 - s_local) * ctrl_wps[k] + s_local * ctrl_wps[k + 1]
            K = K_list[k]
        else:
            qpos_ref = wps[-1]
            ctrl_ref = ctrl_wps[-1]
            K = K1

        for _ in range(n_sub):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_ref, data.qpos)
            dv = data.qvel - 0.0
            x = np.concatenate([dq, dv])
            u = ctrl_ref - K @ x
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        tilt = common.tilt_deg(data.qpos)
        fz_R = foot_fz(model, data, foot_R_id)
        if verbose and tick % 10 == 0:
            print(f"  [{tag}] t={data.time:5.2f} tilt={tilt:6.2f} fz_빨강={fz_R:7.1f}")
        if liftoff_t is None and fz_R < 0.5 and t > 0.3 * T_transition:
            liftoff_t = data.time
        if tilt > 55 and fell_t is None:
            fell_t = data.time
            break
    post = None if (liftoff_t is None or fell_t is None) else fell_t - liftoff_t
    print(f"[{tag}] liftoff={liftoff_t} fell={fell_t} post_survival={post}")
    return liftoff_t, fell_t


def main():
    print("=== B: 폭0.40+발오프셋, kp/kv 기본(x1) ===")
    run_variant("B_widefeet_kpkv1", widen=True, feet=True, kp_kv_scale=1.0,
                eq_path="best_one_leg_qpos_current_gn.npy", verbose=True)
    print("=== D: 좁은 폭(원래), kp/kv x3 ===")
    run_variant("D_narrow_kpkv3", widen=False, feet=False, kp_kv_scale=3.0,
                eq_path="best_one_leg_qpos_narrow_gn.npy", verbose=True)


if __name__ == "__main__":
    main()
