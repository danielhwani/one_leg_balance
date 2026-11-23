"""tvlqr_step_v4_faster.py
---------------------------
tvlqr_step_v2.py(3.58초, T_transition=4.0초)의 실패는 t=3.2~3.4초
(전체의 80~85% 지점, 스윙이 거의 끝나갈 때)에 일어난다. 같은
스케줄 비율(리언 35%)을 유지한 채 전체 전이 시간만 줄여서(4.0->2.5초)
실패 시점보다 먼저 전이를 끝낼 수 있는지 테스트한다.
"""
import math

import numpy as np
import mujoco

import common
from find_one_leg_equilibrium_v2 import solve_torso_for_lean, mat2quat


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


def main():
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    qpos_eq1 = np.load("best_one_leg_qpos_v5.npy")

    data0 = mujoco.MjData(model)
    data0.qpos[:] = qpos_eq0
    mujoco.mj_forward(model, data0)
    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    target_foot_pos = data0.xpos[foot_L_id].copy()
    target_foot_mat = data0.xmat[foot_L_id].copy().reshape(3, 3)

    def jq(name):
        return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]

    q_abduct_L = jq("hip_abduct_L")  # 스탠스 다리 리언(theta1)
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

    # 진단 결과(tvlqr_diagnose.py, 2026-11-23): waist_roll/스탠스
    # 리언(hip_abduct_L)이 실제로는 자기 목표보다 훨씬 앞서(전반부에)
    # 다 끝내버리려 하고, 그게 균형에 도움이 됐다(억지로 못 하게
    # 막으니 더 나빠졌다). 그 자연스러운 경향을 궤적 스케줄 자체에
    # 반영한다 - 리언은 전반부(앞 35%)에 빨리 끝내고, 스윙 다리는
    # 원래대로 전체 구간에 걸쳐 진행.
    LEAN_FRAC = 0.35

    def build_qpos_at(s_global):
        lean_s = smoothstep(min(1.0, s_global / LEAN_FRAC))
        swing_s = smoothstep(s_global)

        theta1 = theta1_final * lean_s
        P0, R0 = solve_torso_for_lean(theta1, theta2, theta3, theta4, theta5,
                                       target_foot_pos, target_foot_mat)
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

    T_transition = 2.5
    n_wp = 40
    wps = [build_qpos_at(i / n_wp) for i in range(n_wp + 1)]
    ctrl_wps = [ctrl_of(q) for q in wps]

    print("웨이포인트별 선형화 중...")
    A_list, B_list = [], []
    for q, c in zip(wps, ctrl_wps):
        A, B = linearize_at(model, q, c, nv, nu)
        A_list.append(A)
        B_list.append(B)
    print(f"{len(wps)}개 웨이포인트 선형화 완료")

    P = Q.copy()
    K_list = [None] * (n_wp + 1)
    A_end, B_end = A_list[-1], B_list[-1]
    for _ in range(200):
        BtP = B_end.T @ P
        K_end = np.linalg.solve(R + BtP @ B_end, BtP @ A_end)
        P = Q + A_end.T @ P @ A_end - A_end.T @ P @ B_end @ K_end
    K_list[n_wp] = K_end
    print("역방향 Riccati 재귀로 K(t) 계산 중...")
    for k in range(n_wp - 1, -1, -1):
        A, B = A_list[k], B_list[k]
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        K_list[k] = K
    print("TV-LQR 게인 계산 완료")

    K1 = K_list[n_wp]
    dt_wp = T_transition / n_wp

    def run(duration_s, hold_s, verbose_every=10, tag=""):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos_eq0
        data.qvel[:] = qvel_eq0
        mujoco.mj_forward(model, data)
        fell = None
        for tick in range(int((duration_s + hold_s) / 0.01)):
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
                qpos_ref = qpos_eq1
                ctrl_ref = ctrl_wps[-1]
                K = K1

            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_ref, data.qpos)
            dv = data.qvel - 0.0
            x = np.concatenate([dq, dv])
            u = ctrl_ref - K @ x
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            for _ in range(2):
                mujoco.mj_step(model, data)

            tilt = common.tilt_deg(data.qpos)
            if tick % verbose_every == 0:
                knee_actual = math.degrees(data.qpos[q_knee_R])
                knee_target = math.degrees(qpos_ref[q_knee_R])
                print(f"  [{tag}] t={data.time:.2f} tilt={tilt:6.2f} "
                      f"|v|={np.linalg.norm(data.qvel[:3]):.4f} "
                      f"knee_actual={knee_actual:5.1f} knee_target={knee_target:5.1f}")
            if tilt > 55:
                fell = data.time
                print(f"[{tag}] 쓰러짐: t={fell:.2f}")
                break
        if fell is None:
            print(f"[{tag}] {duration_s + hold_s}초 동안 버팀!")
        return fell

    print()
    print("=== TV-LQR v2 (스윙 다리 전반부에 다 끝냄) ===")
    run(T_transition, 5.0, verbose_every=10, tag="tvlqr_v2")

    return wps, ctrl_wps, K_list, T_transition, n_wp, qpos_eq0, qvel_eq0, qpos_eq1, Q, R


if __name__ == "__main__":
    main()
