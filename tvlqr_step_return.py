"""tvlqr_step_return.py
------------------------
외발서기는 최종목표가 아니라 중간단계였다 - 진짜 목표는 "발을
들었다가 원래 위치로 다시 내려놓고 두 발로 안정적으로 서는 것"
(스텝의 가장 단순한 버전: 제자리 발 떼기+다시 딛기). 전체 사이클:

  [양발 서기] --(들기, tvlqr_lift_hold_gain.py와 동일)--> [외발 지지]
  --(짧게 유지, 검증된 1.74초 한도 안에서)--> [내리기, 같은 웨이포인트
  테이블을 역순으로]--> [착지] --(두 발 K0로 전환)--> [양발 서기]

내리기는 들기와 별도로 새로 설계하지 않고, 이미 만들어둔 wps/K_list/
ctrl_wps 테이블을 **그대로 역순으로 재생**한다 - LQR 게인은 특정
시간방향에 종속되지 않고 "그 선형화 지점 근처에서 유효한 피드백"
이므로 역방향으로 읽어도 수학적으로 똑같이 유효하다.
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


def run(T_transition=8.0, LEAN_FRAC=0.70, hold_after_liftoff=0.4, T_descend=4.0,
        ankle_r_scale=0.0003, orient_q_scale=50.0, swing_r_scale=0.3,
        duration_s=25.0, verbose=False, tag="",
        eq_path="best_one_leg_qpos_current_gn.npy", use_hold_gain=True,
        descend_trigger_s=None, target_scale=1.0):
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
    theta1_final = qpos_eq1[q_abduct_L] * target_scale
    waist_roll_0 = qpos_eq0[q_waist_roll]
    waist_roll_final = waist_roll_0 + (qpos_eq1[q_waist_roll] - waist_roll_0) * target_scale
    waist_pitch_0 = qpos_eq0[q_waist_pitch]
    waist_pitch_final = waist_pitch_0 + (qpos_eq1[q_waist_pitch] - waist_pitch_0) * target_scale
    abduct_R_final = qpos_eq1[q_abduct_R] * target_scale
    hip_R_final = qpos_eq1[q_hip_R] * target_scale
    knee_R_final = qpos_eq1[q_knee_R] * target_scale

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
    dt_wp_lift = T_transition / n_wp
    dt_wp_desc = T_descend / n_wp

    def lookup(s):
        s = max(0.0, min(1.0, s))
        k = min(n_wp - 1, int(s * n_wp))
        s_local = s * n_wp - k
        qpos_ref = (1 - s_local) * wps[k] + s_local * wps[k + 1]
        qpos_ref[3:7] = wps[k][3:7] * (1 - s_local) + wps[k + 1][3:7] * s_local
        n = np.linalg.norm(qpos_ref[3:7])
        if n > 1e-9:
            qpos_ref[3:7] /= n
        ctrl_ref = (1 - s_local) * ctrl_wps[k] + s_local * ctrl_wps[k + 1]
        K = (1 - s_local) * K_list[k] + s_local * K_list[k + 1]
        return qpos_ref, ctrl_ref, K

    # 유지(hold) 전용 K
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

    # 양발 복귀용 K0 (원래/검증된 두 발 서있기 LQR, 전이용 Q/R 아님)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q0, R0c = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q0, R0c)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    phase = "lift"
    s_at_liftoff = [1.0]
    liftoff_t = None
    hold_start_t = None
    descend_start_t = None
    landed_t = None
    restand_t = None
    fell_t = None

    for tick in range(int(duration_s / 0.01)):
        t = data.time
        fz_R_check = foot_fz(model, data, foot_R_id)

        if phase == "lift":
            s = min(1.0, t / T_transition)
            if fz_R_check < 0.5 and t > 0.3 * T_transition:
                liftoff_t = t
                if use_hold_gain:
                    phase = "hold"
                    hold_start_t = t
                    if verbose:
                        print(f"  [{tag}] >>> 이륙, 유지 시작 t={t:.3f}")
                else:
                    phase = "descend"
                    descend_start_t = t
                    s_at_liftoff[0] = s if descend_trigger_s is None else descend_trigger_s
                    if verbose:
                        print(f"  [{tag}] >>> 이륙, 바로 내리기 시작 t={t:.3f} (s={s_at_liftoff[0]:.3f})")
            qpos_ref, ctrl_ref, _ = lookup(s)
            K = K_list[min(n_wp - 1, int(s * n_wp))]
        elif phase == "hold":
            if t - hold_start_t >= hold_after_liftoff:
                phase = "descend"
                descend_start_t = t
                s_at_liftoff[0] = 1.0
                if verbose:
                    print(f"  [{tag}] >>> 내리기 시작 t={t:.3f}")
            qpos_ref = wps[-1]
            ctrl_ref = ctrl_wps[-1]
            K = K_hold
        elif phase == "descend":
            s0 = s_at_liftoff[0]
            s = max(0.0, s0 * (1.0 - (t - descend_start_t) / T_descend))
            if landed_t is None and fz_R_check > 0.3 * fz_R_nominal:
                landed_t = t
                if verbose:
                    print(f"  [{tag}] >>> 착지! t={t:.3f} (s={s:.3f})")
            if s <= 0.0:
                phase = "restand"
                restand_t = t
                if verbose:
                    print(f"  [{tag}] >>> 양발 K0로 복귀 t={t:.3f}")
            qpos_ref, ctrl_ref, K = lookup(s)
        else:  # restand
            qpos_ref = qpos_eq0
            ctrl_ref = np.zeros(nu)
            K = K0

        for _ in range(n_sub):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_ref, data.qpos)
            if phase == "restand":
                dv = data.qvel - qvel_eq0
            else:
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
            print(f"  [{tag}] t={data.time:5.2f} phase={phase:8s} tilt={tilt:6.2f} "
                  f"fz_빨강={fz_R:7.1f} fz_파랑={fz_L:7.1f}")
        if tilt > 55 and fell_t is None:
            fell_t = data.time
            if verbose:
                print(f"  [{tag}] >>> 쓰러짐! t={fell_t:.3f}")
            break

    result = dict(liftoff=liftoff_t, landed=landed_t, restand=restand_t, fell=fell_t,
                   final_phase=phase, final_tilt=common.tilt_deg(data.qpos))
    print(f"[{tag}] {result}")
    return result


def main():
    print("=== 들기 -> 짧은 유지(0.4s) -> 내리기(4s) -> 양발 복귀 ===")
    run(verbose=True, tag="full_cycle", duration_s=25.0)


if __name__ == "__main__":
    main()
