"""tvlqr_lift_hold_gain.py
---------------------------
tvlqr_lift_ablation.py 최고 설정(T=6.0, LEAN_FRAC=0.7, 넓은 폭+발,
kp/kv x1)의 생생한 trace를 보면: 이륙(t=1.82) 후에도 t=2.0까지는
tilt<1'로 거의 완벽하게 버티다가 t~2.1-2.2s부터 갑자기 지수적으로
발산한다. 지금 "유지용" 게인(K1)은 전이 스케줄의 마지막
웨이포인트에서 전이용 Q/R(스윙다리 추종 가중치 x30)을 그대로 이어받은
것일 뿐 - "외발로 오래 버티기"에 맞춰 따로 설계된 적이 없다.

이번엔 두 가지를 분리해서 테스트한다:
1. 전환 시점을 시계(T_transition)가 아니라 실제 접촉력 이벤트
   (fz_빨강<0.5)로 맞춘다 - 지금은 실제 이륙이 스케줄보다 훨씬 빨리
   일어나서(t=1.82 vs T=6.0) 그 사이엔 여전히 "전이 중" 게인을 쓰고
   있었다.
2. 그 전환 시점에 쓰는 게인을 전이용 Q/R이 아니라, 발목
   (ankle_roll_L/ankle_pitch_L, 스탠스 다리)에 추종/제어 권한을 더
   준 "유지 전용" Q/R로 새로 DARE를 풀어서 만든다.
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
        switch_mode="event", verbose=False, tag="", hip_r_scale=1.0, waist_r_scale=1.0,
        blend_T=0.0, switch_fz_threshold=0.5, relax_ctrl_limits=False,
        swing_r_scale=1.0, knee_l_r_scale=1.0, angvel_q_scale=1.0, relin_period=0.0,
        eq_path="best_one_leg_qpos_current_gn.npy"):
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
        # hip_R/L(±35)과 knee_R/L(0~90)은 하드웨어 range가 아니라 코드가
        # 임의로 30도로 잘라놓은 것 - 오늘 어깨에서 발견한 것과 같은 함정.
        # 외발 평형점 자체가 knee_R=35도를 요구하는데 30도로 잘리고 있었다.
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
    dt_wp = T_transition / n_wp

    # --- 유지(hold) 전용 K: 전이 Q/R이 아니라 발목 권한+상체 자세+CoM
    # 가중치를 올린 별도 Q_hold/R_hold로 최종 평형점에서 새로 DARE ---
    Q_hold, R_hold = common.build_cost_two_leg(model)
    Q_hold[0:3, 0:3] *= com_q_scale          # CoM(자유관절 위치) 추종 강화
    Q_hold[3:6, 3:6] *= orient_q_scale       # 상체 자세(기울기) 추종 강화
    Q_hold[nv + 3:nv + 6, nv + 3:nv + 6] *= angvel_q_scale  # 각속도 추종 강화
    a_ankle_roll_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_roll_L")
    a_ankle_pitch_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_pitch_L")
    R_hold[a_ankle_roll_L, a_ankle_roll_L] *= ankle_r_scale
    R_hold[a_ankle_pitch_L, a_ankle_pitch_L] *= ankle_r_scale
    a_hip_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_waist_roll = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_waist_roll")
    R_hold[a_hip_abduct_L, a_hip_abduct_L] *= hip_r_scale
    R_hold[a_waist_roll, a_waist_roll] *= waist_r_scale
    for jname in ["hip_abduct_R", "hip_R", "knee_R"]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jname}")
        R_hold[aid, aid] *= swing_r_scale
    a_knee_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_L")
    R_hold[a_knee_L, a_knee_L] *= knee_l_r_scale
    K_hold = common.solve_dare(A_end, B_end, Q_hold, R_hold)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    liftoff_t = None
    fell_t = None
    switch_t = None
    qpos_at_switch = None
    ctrl_at_switch = None
    K_at_switch = None
    last_relin_t = None
    for tick in range(int((T_transition + hold_s) / 0.01)):
        t = data.time
        fz_R_check = foot_fz(model, data, foot_R_id)
        if switch_mode == "event":
            do_switch = (switch_t is not None) or (fz_R_check < switch_fz_threshold and t > 0.3 * T_transition)
        else:
            do_switch = t >= T_transition

        if not do_switch:
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
            if switch_t is None:
                switch_t = t
                qpos_at_switch = qpos_ref.copy()
                ctrl_at_switch = ctrl_ref.copy()
                K_at_switch = K.copy()
                if verbose:
                    print(f"  [{tag}] >>> 유지 게인으로 전환 t={t:.3f}")
            if blend_T > 0.0:
                s_blend = min(1.0, (t - switch_t) / blend_T)
            else:
                s_blend = 1.0
            qpos_ref = (1 - s_blend) * qpos_at_switch + s_blend * wps[-1]
            qpos_ref[3:7] = qpos_at_switch[3:7] * (1 - s_blend) + wps[-1][3:7] * s_blend
            n = np.linalg.norm(qpos_ref[3:7])
            if n > 1e-9:
                qpos_ref[3:7] /= n
            ctrl_ref = (1 - s_blend) * ctrl_at_switch + s_blend * ctrl_wps[-1]
            K = (1 - s_blend) * K_at_switch + s_blend * K_hold

            if relin_period > 0.0 and (last_relin_t is None or t - last_relin_t >= relin_period):
                last_relin_t = t
                A_cur, B_cur = linearize_at(model, data.qpos.copy(), data.ctrl.copy(), nv, nu)
                try:
                    K_hold = common.solve_dare(A_cur, B_cur, Q_hold, R_hold)
                except np.linalg.LinAlgError:
                    pass
                K = K_hold

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
    if fell_t is None:
        print(f"[{tag}] {T_transition+hold_s}초 동안 버팀!")
    post = None if (liftoff_t is None or fell_t is None) else fell_t - liftoff_t
    print(f"[{tag}] liftoff={liftoff_t} fell={fell_t} post_survival={post}")
    return liftoff_t, fell_t


def main():
    print("=== 기본(전환 없음과 동일 효과, ankle_r=1/orient=1/com=1) ===")
    run(6.0, 0.7, 8.0, ankle_r_scale=1.0, orient_q_scale=1.0, com_q_scale=1.0,
        switch_mode="event", verbose=True, tag="baseline_event")

    print("=== 발목 R 낮춤(ankle_r_scale) 스윕 ===")
    for ar in [0.3, 0.1, 0.03, 0.01]:
        run(6.0, 0.7, 8.0, ankle_r_scale=ar, orient_q_scale=1.0, com_q_scale=1.0,
            switch_mode="event", verbose=False, tag=f"ankle_r={ar}")

    print("=== 상체 자세 Q 올림(orient_q_scale) 스윕 ===")
    for oq in [2.0, 5.0, 10.0]:
        run(6.0, 0.7, 8.0, ankle_r_scale=1.0, orient_q_scale=oq, com_q_scale=1.0,
            switch_mode="event", verbose=False, tag=f"orient_q={oq}")

    print("=== CoM Q 올림(com_q_scale) 스윕 ===")
    for cq in [2.0, 5.0, 10.0]:
        run(6.0, 0.7, 8.0, ankle_r_scale=1.0, orient_q_scale=1.0, com_q_scale=cq,
            switch_mode="event", verbose=False, tag=f"com_q={cq}")


if __name__ == "__main__":
    main()
