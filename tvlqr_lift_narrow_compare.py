"""tvlqr_lift_narrow_compare.py
--------------------------------
tvlqr_lift_v1.py가 "현재 구조"(골반폭 0.40 + 발오프셋 + kp/kv x3 +
dt=0.0005)에서 생존시간이 0.5초대로 뚝 떨어졌다 - 이게 내 구현
버그인지, 아니면 그 구조 변경들(원래 두발 push 강건성을 위한 것)이
외발 들기엔 오히려 손해인지 확인하기 위해, 원래(좁은 골반, 기본
kp/kv, 기본 dt) 설정으로 똑같은 TV-LQR을 돌려 비교한다.
"""
import math

import numpy as np
import mujoco

import common
from find_one_leg_equilibrium_v2 import mat2quat
from find_one_leg_eq_current import solve_torso_for_lean_general
from tvlqr_lift_v1 import smoothstep, linearize_at, foot_fz, SHOULDER_TARGET_DEG


def main():
    model = common.load_model()  # 원래(좁은) 설정 그대로
    nv, nu = model.nv, model.nu
    n_sub = round(0.01 / model.opt.timestep)
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    qpos_eq1 = np.load("best_one_leg_qpos_narrow.npy")

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
    q_shoulder_L = jq("shoulder_abduct_L")

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
    shoulder_L_final = math.radians(SHOULDER_TARGET_DEG)

    LEAN_FRAC = 0.35

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
        qpos[q_shoulder_L] = shoulder_L_final * lean_s
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
    a_sh_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_shoulder_abduct_L")
    a_sh_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_shoulder_abduct_R")
    ctrl_limits[a_sh_L] = math.radians(60.0)
    ctrl_limits[a_sh_R] = math.radians(60.0)

    swing_joint_names = ["hip_abduct_R", "hip_R", "knee_R"]
    SWING_TRACK_WEIGHT = 30.0
    for jname in swing_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dofadr = model.jnt_dofadr[jid]
        Q[dofadr, dofadr] *= SWING_TRACK_WEIGHT
        Q[nv + dofadr, nv + dofadr] *= SWING_TRACK_WEIGHT

    def build_and_run(T_transition, hold_s, n_wp=40, verbose=False, tag=""):
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
                fz_L = foot_fz(model, data, foot_L_id)
                print(f"  [{tag}] t={data.time:5.2f} tilt={tilt:6.2f} "
                      f"fz_빨강={fz_R:7.1f} fz_파랑={fz_L:7.1f}")
            if liftoff_t is None and fz_R < 0.5 and t > 0.3 * T_transition:
                liftoff_t = data.time
                if verbose:
                    print(f"  [{tag}] >>> 이륙! t={liftoff_t:.3f}")
            if tilt > 55 and fell_t is None:
                fell_t = data.time
                if verbose:
                    print(f"  [{tag}] >>> 쓰러짐! t={fell_t:.3f}")
                break
        return liftoff_t, fell_t

    print("=== TV-LQR 외발 들어올리기 (원래/좁은 설정, 비교용) ===")
    for T_transition in [3.0, 4.0, 5.0, 6.0]:
        liftoff, fell = build_and_run(T_transition, 10.0, verbose=(T_transition == 4.0), tag=f"T={T_transition}")
        post = None if (liftoff is None or fell is None) else fell - liftoff
        status = "생존" if fell is None else f"쓰러짐@{fell:.2f}"
        print(f"T_transition={T_transition}: liftoff={liftoff} fell={fell} post_survival={post} [{status}]")


if __name__ == "__main__":
    main()
