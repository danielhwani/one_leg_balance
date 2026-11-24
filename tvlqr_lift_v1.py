"""tvlqr_lift_v1.py
--------------------
오늘 하루 종일 겪은 문제: "체중이동 피드포워드"(목표=두발평형점 K0)와
"CP 피드백"(목표=한발지지 상태)을 그냥 더해서 쓰니 서로 반대 방향으로
계속 부딪혔다(K0가 저항으로 작용). 사용자 결론: "하나의 목표를 향한
하나의 제어기"로 만들어야 한다.

tvlqr_step_v2.py가 이미 그 구조(참조궤적 x_ref(t) + 시간가변 K(t),
u = ctrl_ref(t) - K(t)@(x-x_ref(t)) 하나의 법칙)였고 실제로 3초 이상
외발 지지를 버틴 적이 있다 - 다만 그건 오늘 바뀐 로봇 구조(정강이 버그
수정, 골반 폭 0.40, 발 오프셋 0.28/0.16, 다리 kp/kv x3, dt=0.0005) 전의
모델이었다. 이번엔 현재 구조 위에서, 그리고 오늘 찾은 "팔 스윙이 CP를
당긴다"는 발견도 별도 피드백 루프가 아니라 **궤적 자체(x_ref(t))의
일부**로 포함시켜서 다시 만든다.

중요: K(t)가 느슨한 스케줄(10ms)로 와도, 실제 피드백 보정
u = ctrl_ref - K@(x-x_ref)의 x는 매 물리 서브스텝(0.0005s)마다
다시 재야 한다 - 안 그러면 kp_kv_timestep_test.py에서 배운 것과
같은 버그(강성 액추에이터를 과소 구동)가 재발한다.
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

GRAVITY = 9.81
SHOULDER_TARGET_DEG = 45.0


def smoothstep(s):
    s = max(0.0, min(1.0, s))
    return 3 * s * s - 2 * s * s * s


def linearize_at(model, qpos, ctrl, nv, nu):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    return A, B


def foot_fz(model, data, body_id):
    total = 0.0
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if b1 != body_id and b2 != body_id:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        frame = c.frame.reshape(3, 3)
        total += (frame.T @ force[:3])[2]
    return total


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
    qpos_eq1 = np.load("best_one_leg_qpos_current.npy")

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

    # 리언(스탠스 다리+허리+팔)은 전반부에 먼저 끝내고, 스윙 다리는
    # 전체 구간에 걸쳐 진행 - tvlqr_step_v2.py에서 검증된 스케줄
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

    # 팔(shoulder)은 하드웨어 한계가 아니라 코드에서 임의로 30도로
    # 잘라놨던 것 뿐이었다(XML에 ctrlrange 없음) - 오늘 발견한 "팔이
    # 포화돼서 못 썼다"는 병목을 없애려고 팔만 관절 한계(60도)까지 허용
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

    print("=== TV-LQR 외발 들어올리기 (현재 로봇 구조: 정강이 수정+폭0.40+발오프셋+kp/kv x3) ===")
    for T_transition in [3.0, 4.0, 5.0, 6.0]:
        liftoff, fell = build_and_run(T_transition, 10.0, verbose=(T_transition == 4.0), tag=f"T={T_transition}")
        post = None if (liftoff is None or fell is None) else fell - liftoff
        status = "생존" if fell is None else f"쓰러짐@{fell:.2f}"
        print(f"T_transition={T_transition}: liftoff={liftoff} fell={fell} post_survival={post} [{status}]")


if __name__ == "__main__":
    main()
