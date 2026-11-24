"""view_tvlqr_lift_final.py
----------------------------
오늘 찾은 최선 설정(tvlqr_lift_hold_gain.py, 이륙후 생존 ~1.74초)을
뷰어로 직접 본다:
  - 넓은 골반(0.40)+넓은 발(0.28/0.16), 다리 kp/kv는 이 전이 구간에서
    x1(x3는 오히려 손해 - kp/kv 스윕으로 확인됨)
  - 전이: T_transition=8.0s, LEAN_FRAC=0.70 (스탠스 다리 리언+허리+팔은
    전반부 70%에 끝내고, 스윙 다리는 전체 구간에 걸쳐 진행)
  - 실제 접촉력 기반 이륙 이벤트에서(시계가 아니라) 전용 "유지(hold)"
    게인으로 전환: 스탠스 발목 R을 0.0003배로 낮추고(발목 전략),
    상체 자세 Q를 50배 올리고, 공중에 뜬 스윙다리에도 약간의 권한
    (R x0.3)을 준다.
  - hip_R/knee_R의 ctrl_limits를 코드상의 임의 30도 캡이 아니라 실제
    관절 range로 완화(평형점 자체의 knee_R=35도 목표가 30도로 잘리고
    있었다).

실행: python view_tvlqr_lift_final.py
창을 닫으면 종료. 빨간 발(오른발)을 들어올리는 전이가 끝까지 잘 되면
노란 공(캡처포인트)/하늘색 공(CoM) 없이 순수 관절 거동만 본다 - 결국
~1.7~1.8초 뒤에는 쓰러진다(아직 완전한 외발 서있기는 아님).
"""
import math
import time

import numpy as np
import mujoco
import mujoco.viewer

import common
from pelvis_width_test import widen_pelvis
from foot_offset_test import set_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv
from find_one_leg_equilibrium_v2 import mat2quat
from find_one_leg_eq_current import solve_torso_for_lean_general
from tvlqr_lift_v1 import smoothstep, linearize_at, foot_fz

# 오늘 찾은 최선 설정
T_TRANSITION = 8.0
LEAN_FRAC = 0.70
ANKLE_R_SCALE = 0.0003
ORIENT_Q_SCALE = 50.0
COM_Q_SCALE = 1.0
SWING_R_SCALE = 0.3
SWITCH_FZ_THRESHOLD = 0.5


def build_model():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.28, 0.16, 0.0)
    scale_kp_kv(model, LEG_ACTUATORS, 1.0)  # 이 전이에선 x3가 오히려 손해
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

    print("웨이포인트 선형화 + TV-LQR 게인 계산 중...")
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
    dt_wp = T_TRANSITION / n_wp

    # 유지(hold) 전용 K: 발목 전략 + 상체 자세 가중 + 스윙다리 보조 권한
    Q_hold, R_hold = common.build_cost_two_leg(model)
    Q_hold[0:3, 0:3] *= COM_Q_SCALE
    Q_hold[3:6, 3:6] *= ORIENT_Q_SCALE
    a_ankle_roll_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_roll_L")
    a_ankle_pitch_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_pitch_L")
    R_hold[a_ankle_roll_L, a_ankle_roll_L] *= ANKLE_R_SCALE
    R_hold[a_ankle_pitch_L, a_ankle_pitch_L] *= ANKLE_R_SCALE
    for jname in ["hip_abduct_R", "hip_R", "knee_R"]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jname}")
        R_hold[aid, aid] *= SWING_R_SCALE
    K_hold = common.solve_dare(A_end, B_end, Q_hold, R_hold)
    print("계산 완료. 뷰어 시작.")

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    state = {"switch_t": None, "qpos_at_switch": None, "ctrl_at_switch": None,
             "liftoff_t": None, "fell_t": None, "last_print": -1.0}

    print(f"설정: T_transition={T_TRANSITION}s, LEAN_FRAC={LEAN_FRAC}, "
          f"ankle_r={ANKLE_R_SCALE}, orient_q={ORIENT_Q_SCALE}, swing_r={SWING_R_SCALE}")
    print("창을 닫으면 종료.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 6.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0.0, 0.0, 1.2]
        while viewer.is_running():
            t = data.time
            fz_R_check = foot_fz(model, data, foot_R_id)
            do_switch = (state["switch_t"] is not None) or \
                (fz_R_check < SWITCH_FZ_THRESHOLD and t > 0.3 * T_TRANSITION)

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
                if state["switch_t"] is None:
                    state["switch_t"] = t
                    print(f"  >>> 유지 게인으로 전환 t={t:.3f}")
                qpos_ref = wps[-1]
                ctrl_ref = ctrl_wps[-1]
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

            viewer.sync()
            time.sleep(max(0.0, 0.01 - 0.0005))

            tilt = common.tilt_deg(data.qpos)
            fz_R = foot_fz(model, data, foot_R_id)
            if state["liftoff_t"] is None and fz_R < 0.5 and t > 0.3 * T_TRANSITION:
                state["liftoff_t"] = data.time
                print(f"  >>> 이륙! t={state['liftoff_t']:.3f}")
            if tilt > 55 and state["fell_t"] is None:
                state["fell_t"] = data.time
                post = (state["fell_t"] - state["liftoff_t"]) if state["liftoff_t"] else None
                print(f"  >>> 쓰러짐! t={state['fell_t']:.3f} "
                      f"(이륙후 생존={post:.2f}s)" if post else f"  >>> 쓰러짐! t={state['fell_t']:.3f}")
            if data.time - state["last_print"] >= 0.3:
                state["last_print"] = data.time
                print(f"  t={data.time:6.2f}s tilt={tilt:5.2f} fz_빨강={fz_R:7.1f}")


if __name__ == "__main__":
    main()
