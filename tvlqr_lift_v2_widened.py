"""tvlqr_lift_v2_widened.py
----------------------------
tvlqr_step_v2.py를 오늘(2026-11-24) 그대로 다시 돌려보니(수정 없이)
t=2.61s에 쓰러졌다 - 기록된 "3.1~3.58s"와 비슷한 정도(정강이 버그
수정이 이 시나리오를 망가뜨리지 않았다는 뜻). v5 평형점 자체도
|qacc_free|=1.818로 완전히 수렴한 게 아니라 원래 그 정도 정밀도였다.

이번엔 정확히 같은 구조(스윙 다리 전반부 집중 스케줄, Q 가중치,
T_transition=4.0, 가우스-뉴턴 평형점)를 오늘 바뀐 구조(골반폭 0.40 +
발 오프셋 0.28/0.16 + 다리 kp/kv x3 + dt=0.0005)에 적용해서, 좁은
설정의 2.61s와 직접 비교한다 - 다른 변수 없이 "구조 확장이 외발
들기에 손해인가"만 본다. 평형점은 find_one_leg_eq_gn.py가 만든
best_one_leg_qpos_current_gn.npy(v5와 같은 abduct=-25/knee=35/hip=10
스윙 조합, 같은 가우스-뉴턴, |qacc_free|=1.815로 v5와 거의 동급 정밀도).
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


def build_model():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.28, 0.16, 0.0)
    scale_kp_kv(model, LEG_ACTUATORS, 3.0)
    model.opt.timestep = 0.0005
    return model


def main():
    model = build_model()
    nv, nu = model.nv, model.nu
    n_sub = round(0.01 / model.opt.timestep)
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    qpos_eq1 = np.load("best_one_leg_qpos_current_gn.npy")

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
                knee_actual = math.degrees(data.qpos[q_knee_R])
                knee_target = math.degrees(qpos_ref[q_knee_R])
                print(f"  [{tag}] t={data.time:5.2f} tilt={tilt:6.2f} "
                      f"knee_actual={knee_actual:5.1f} knee_target={knee_target:5.1f} "
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
        if fell_t is None:
            print(f"[{tag}] {T_transition+hold_s}초 동안 버팀!")
        return liftoff_t, fell_t

    print("=== TV-LQR 외발 들기 (v5 스윙 조합 abduct=-25/knee=35/hip=10, 현재/넓은 구조) ===")
    for T_transition in [4.0]:
        liftoff, fell = build_and_run(T_transition, 5.0, verbose=True, tag=f"T={T_transition}")
        post = None if (liftoff is None or fell is None) else fell - liftoff
        print(f"T_transition={T_transition}: liftoff={liftoff} fell={fell} post_survival={post}")


if __name__ == "__main__":
    main()
