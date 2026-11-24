"""tvlqr_lift_state_phase.py
-----------------------------
tvlqr_lift_hold_gain.py(최선 1.74s)는 전이 스케줄을 시계(t/T_transition)
로 색인했다 - 실제 이륙은 그 시계가 가리키는 "진행률"보다 훨씬 빨리
일어나서(s_global=0.3일 때 이미 완전히 떠있었음), 그 사이엔 현실과
안 맞는 웨이포인트/게인을 쓰고 있었다. 이게 "모순"(두 제어기가 다른
목표로 싸움)은 아니지만, "계획과 현실의 시점 불일치"라는 별도의
오차 원인이었다.

이번엔 같은 웨이포인트/게인 테이블(K_list, wps)을 그대로 쓰되,
그걸 색인하는 진행률 s_global 자체를 **시계가 아니라 실측 상태**로
다시 정의한다:
  s_contact = 1 - fz_빨강/fz_빨강_정지시 (접촉력이 실제로 얼마나
              빠졌는지 - 이게 진짜 "얼마나 외발로 전환됐는지"다)
  s_global = ratchet(max(s_time, s_contact))  (한 번 전진하면 절대
              후퇴하지 않는 래칫 - 시계보다 빨리 빠지면 즉시 앞당겨
              따라가고, 느리면 원래 시계 스케줄을 그대로 따름)
여전히 선형(TV-LQR) 제어 그대로다 - 새 알고리즘이 아니라 같은
테이블을 "언제" 읽을지만 바꾼 것.
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
    scale_kp_kv(model, LEG_ACTUATORS, 1.0)
    model.opt.timestep = 0.0005
    return model


def run(T_transition, LEAN_FRAC, hold_s, ankle_r_scale, orient_q_scale, com_q_scale,
        verbose=False, tag="", swing_r_scale=1.0, relax_ctrl_limits=True,
        switch_s_threshold=0.99, eq_path="best_one_leg_qpos_current_gn.npy",
        use_state_phase=True, filter_tau=0.15, switch_mode_event=False):
    model = build_model()
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
    fz_R_nominal = foot_fz(model, data0, foot_R_id)

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
    if relax_ctrl_limits:
        for jname in ["hip_R", "hip_L", "knee_R", "knee_L"]:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jname}")
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            jrange = model.jnt_range[jid]
            ctrl_limits[aid] = max(abs(jrange[0]), abs(jrange[1]))
    for jname in ["hip_abduct_R", "hip_R", "knee_R"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dofadr = model.jnt_dofadr[jid]
        Q[dofadr, dofadr] *= 30.0
        Q[nv + dofadr, nv + dofadr] *= 30.0

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

    Q_hold, R_hold = common.build_cost_two_leg(model)
    Q_hold[3:6, 3:6] *= orient_q_scale
    a_ankle_roll_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_roll_L")
    a_ankle_pitch_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_pitch_L")
    R_hold[a_ankle_roll_L, a_ankle_roll_L] *= ankle_r_scale
    R_hold[a_ankle_pitch_L, a_ankle_pitch_L] *= ankle_r_scale
    for jname in ["hip_abduct_R", "hip_R", "knee_R"]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jname}")
        R_hold[aid, aid] *= swing_r_scale
    K_hold = common.solve_dare(A_end, B_end, Q_hold, R_hold)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    liftoff_t = None
    fell_t = None
    switched = False
    s_ratchet = 0.0
    fz_filtered = fz_R_nominal
    alpha = 0.01 / max(filter_tau, 1e-6)  # 1차 저역통과(EMA), dt=0.01s 기준
    for tick in range(int((T_transition + hold_s) / 0.01)):
        t = data.time
        fz_R_check = foot_fz(model, data, foot_R_id)
        fz_filtered += alpha * (fz_R_check - fz_filtered)

        s_time = min(1.0, t / T_transition)
        if use_state_phase:
            # 접촉력 원신호는 체중이동 중에도 80~190N 사이로 출렁여서
            # (노이즈로 래칫이 너무 일찍 튐) 바로 못 쓴다 - 저역통과
            # 필터를 걸어 "진짜 추세"만 본다.
            s_contact = max(0.0, min(1.0, 1.0 - fz_filtered / fz_R_nominal))
            s_candidate = max(s_time, s_contact)
        else:
            s_candidate = s_time
        s_ratchet = max(s_ratchet, s_candidate)
        s_global = s_ratchet
        event_switch = fz_R_check < 0.5 and t > 0.3 * T_transition

        if s_global >= switch_s_threshold or (switch_mode_event and (switched or event_switch)):
            if not switched and verbose:
                print(f"  [{tag}] >>> 유지 게인으로 전환 t={t:.3f} (s_global={s_global:.3f})")
            switched = True
            qpos_ref = wps[-1]
            ctrl_ref = ctrl_wps[-1]
            K = K_hold
        else:
            k = min(n_wp - 1, int(s_global * n_wp))
            s_local = s_global * n_wp - k
            qpos_ref = (1 - s_local) * wps[k] + s_local * wps[k + 1]
            qpos_ref[3:7] = wps[k][3:7] * (1 - s_local) + wps[k + 1][3:7] * s_local
            n = np.linalg.norm(qpos_ref[3:7])
            if n > 1e-9:
                qpos_ref[3:7] /= n
            ctrl_ref = (1 - s_local) * ctrl_wps[k] + s_local * ctrl_wps[k + 1]
            K = K_list[k]

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
            print(f"  [{tag}] t={data.time:5.2f} tilt={tilt:6.2f} s_global={s_global:.3f} fz_빨강={fz_R:7.1f}")
        if liftoff_t is None and fz_R < 0.5 and t > 0.3 * T_transition:
            liftoff_t = data.time
        if tilt > 55 and fell_t is None:
            fell_t = data.time
            break
    if fell_t is None:
        print(f"[{tag}] {T_transition+hold_s}초 동안 버팀!")
    post = None if (liftoff_t is None or fell_t is None) else fell_t - liftoff_t
    print(f"[{tag}] liftoff={liftoff_t} fell={fell_t} post_survival={post}")
    return liftoff_t, fell_t


def main():
    print("=== 비교: 시간기반 vs 상태기반 phase (동일 최선 게인) ===")
    run(8.0, 0.7, 12.0, ankle_r_scale=0.0003, orient_q_scale=50.0, com_q_scale=1.0,
        verbose=False, tag="time_based", swing_r_scale=0.3, use_state_phase=False)
    run(8.0, 0.7, 12.0, ankle_r_scale=0.0003, orient_q_scale=50.0, com_q_scale=1.0,
        verbose=True, tag="state_based", swing_r_scale=0.3, use_state_phase=True)


if __name__ == "__main__":
    main()
