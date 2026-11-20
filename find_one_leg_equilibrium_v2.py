"""find_one_leg_equilibrium_v2.py
----------------------------------
1차 시도(그리드 + 상수 ctrl로 3초 시뮬레이션)는 잘못된 기준으로
평가했다 - "오픈루프로 3초 버티는가"를 봤는데, 두 발 서있기의
qpos_eq0조차 ctrl=0(피드백 없음)이면 4.78초 안에 쓰러진다(원본
프로젝트 1일차 교훈). 평형점은 "열린 루프로 안정"이 아니라
"그 점에서 가속도=0"이면 충분하다 - 그 다음 LQR이 그 점 주변의
작은 편차를 안정화한다.

이번엔 역기구학(analytic FK)으로 제대로 푼다:
- 오른다리(hip_abduct_R=θ1만 바꾸고 나머지 오른다리 관절은 서있기
  값 그대로 유지)로 다리를 기울이면, 발(foot_R)이 원래 서있던
  위치/방향에 그대로 있으려면 **몸통(torso, free joint)이 그만큼
  반대로 기울어야 한다** - 이걸 체인을 직접 행렬로 풀어서 몸통의
  필요한 pos/quat을 계산한다 (수치 최적화나 시뮬레이션 없이 한 번에).
- θ1을 스윕하면서 "이 자세에서 ctrl=qpos(관절 피드백 에러 0)로
  mj_forward 했을 때 몸통(자유관절)의 가속도가 가장 작아지는 지점"을
  찾는다 - 그게 몸 전체 CoM이 오른발 위로 정확히 오는 지점이다.
"""
import math

import numpy as np
import mujoco

import common


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def solve_torso_for_lean(theta1, theta2, theta3, theta4, theta5,
                          target_foot_pos, target_foot_mat):
    R_b = Rx(theta1) @ Ry(theta2)
    R_c = R_b @ Ry(theta3)
    R_e = R_c @ Rx(theta4) @ Ry(theta5)
    p_chain = np.array([0.0, 0.25, -0.7]) + R_b @ np.array([0.0, 0.0, -0.6]) \
        + R_c @ np.array([0.0, 0.0, -0.6])
    R0 = target_foot_mat @ R_e.T
    P0 = target_foot_pos - R0 @ p_chain
    return P0, R0


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


def main():
    model = common.load_model()
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)

    data0 = mujoco.MjData(model)
    data0.qpos[:] = qpos_eq0
    data0.qvel[:] = 0.0
    mujoco.mj_forward(model, data0)

    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    target_foot_pos = data0.xpos[foot_R_id].copy()
    target_foot_mat = data0.xmat[foot_R_id].copy().reshape(3, 3)

    # 왼발을 충돌 비활성화 - 땅과 절대 접촉하지 않게 해서 순수 "오른발
    # 하나로만 지지"하는 조건을 정확히 만든다 (무릎 각도로 클리어런스
    # 맞추는 기하학적 문제를 아예 없앤다)
    foot_L_geom = model.body_geomadr[foot_L_id]
    model.geom_contype[foot_L_geom] = 0
    model.geom_conaffinity[foot_L_geom] = 0

    def jq(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid]

    q_abduct_R = jq("hip_abduct_R")
    q_hip_R = jq("hip_R")
    q_knee_R = jq("knee_R")
    q_ankle_roll_R = jq("ankle_roll_R")
    q_ankle_pitch_R = jq("ankle_pitch_R")
    q_abduct_L = jq("hip_abduct_L")
    q_hip_L = jq("hip_L")
    q_knee_L = jq("knee_L")
    q_ankle_roll_L = jq("ankle_roll_L")
    q_ankle_pitch_L = jq("ankle_pitch_L")

    theta2 = qpos_eq0[q_hip_R]
    theta3 = qpos_eq0[q_knee_R]
    theta4 = qpos_eq0[q_ankle_roll_R]
    theta5 = qpos_eq0[q_ankle_pitch_R]

    # sanity check: theta1=0 (lean 없음)이면 원래 standing qpos와 같아야 함
    P0_check, R0_check = solve_torso_for_lean(0.0, theta2, theta3, theta4, theta5,
                                               target_foot_pos, target_foot_mat)
    q_check = mat2quat(R0_check)
    print("sanity check (theta1=0): P0 vs 실제 torso pos",
          P0_check, qpos_eq0[0:3])
    print("  quat:", q_check, "vs", qpos_eq0[3:7])

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    q_waist_roll = jq("waist_roll")
    q_waist_pitch = jq("waist_pitch")
    waist_roll_0 = qpos_eq0[q_waist_roll]
    waist_pitch_0 = qpos_eq0[q_waist_pitch]

    total_weight_N = sum(model.body_mass) * 9.81
    print(f"\n전체 몸무게(N): {total_weight_N:.2f} (한 발 평형이면 fz_R이 이 값 근처여야 함)")

    def build_qpos(theta1_deg, waist_roll_delta_deg, waist_pitch_delta_deg, knee_L_deg, hip_L_deg):
        theta1 = math.radians(theta1_deg)
        P0, R0 = solve_torso_for_lean(theta1, theta2, theta3, theta4, theta5,
                                       target_foot_pos, target_foot_mat)
        quat0 = mat2quat(R0)
        qpos = qpos_eq0.copy()
        qpos[0:3] = P0
        qpos[3:7] = quat0
        qpos[q_abduct_R] = theta1
        qpos[q_waist_roll] = waist_roll_0 + math.radians(waist_roll_delta_deg)
        qpos[q_waist_pitch] = waist_pitch_0 + math.radians(waist_pitch_delta_deg)
        qpos[q_knee_L] = math.radians(knee_L_deg)
        qpos[q_hip_L] = math.radians(hip_L_deg)
        qpos[q_ankle_pitch_L] = 0.0
        qpos[q_ankle_roll_L] = 0.0
        return qpos

    def eval_qpos(qpos):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        for i in range(model.nu):
            jid = model.actuator_trnid[i, 0]
            qadr = model.jnt_qposadr[jid]
            data.ctrl[i] = qpos[qadr]
        mujoco.mj_forward(model, data)
        qacc_free = data.qacc[0:6].copy()
        fz_L = foot_fz(model, data, foot_L_id)
        fz_R = foot_fz(model, data, foot_R_id)
        return qacc_free, fz_L, fz_R

    print()
    print("=== 1차: theta1 스윕 (waist_roll=0, waist_pitch=0) ===")
    best = None
    for theta1_deg in np.arange(-30, -5, 1.0):
        qpos = build_qpos(theta1_deg, 0.0, 0.0, 30, 0)
        qacc_free, fz_L, fz_R = eval_qpos(qpos)
        acc_norm = np.linalg.norm(qacc_free)
        if best is None or acc_norm < best[0]:
            best = (acc_norm, theta1_deg, 0.0, 0.0, fz_L, fz_R, qacc_free.copy())
    print(f"1차 최적: theta1={best[1]} |qacc_free|={best[0]:.3f} fz_R={best[5]:.2f}")

    print()
    print("=== 2차: best theta1 주변 + waist_roll 2D 스윕 ===")
    theta1_center = best[1]
    best2 = None
    for theta1_deg in np.arange(theta1_center - 6, theta1_center + 6, 1.0):
        for waist_roll_delta in np.arange(-20, 21, 2.0):
            qpos = build_qpos(theta1_deg, waist_roll_delta, 0.0, 30, 0)
            qacc_free, fz_L, fz_R = eval_qpos(qpos)
            acc_norm = np.linalg.norm(qacc_free)
            if best2 is None or acc_norm < best2[0]:
                best2 = (acc_norm, theta1_deg, waist_roll_delta, 0.0, fz_L, fz_R, qacc_free.copy())
    print(f"2차 최적: theta1={best2[1]} waist_roll_delta={best2[2]} "
          f"|qacc_free|={best2[0]:.3f} fz_R={best2[5]:.2f}")
    print(f"  qacc_free 분해(lin xyz, ang xyz): {best2[6]}")

    print()
    print("=== 3차: best2 주변 + waist_pitch 3D 스윕 ===")
    theta1_c, wroll_c = best2[1], best2[2]
    best3 = None
    for theta1_deg in np.arange(theta1_c - 4, theta1_c + 4, 1.0):
        for waist_roll_delta in np.arange(wroll_c - 6, wroll_c + 6, 2.0):
            for waist_pitch_delta in np.arange(-25, 26, 2.0):
                qpos = build_qpos(theta1_deg, waist_roll_delta, waist_pitch_delta, 30, 0)
                qacc_free, fz_L, fz_R = eval_qpos(qpos)
                acc_norm = np.linalg.norm(qacc_free)
                if best3 is None or acc_norm < best3[0]:
                    best3 = (acc_norm, theta1_deg, waist_roll_delta, waist_pitch_delta,
                             fz_L, fz_R, qacc_free.copy(), qpos.copy())
    print(f"3차 최적: theta1={best3[1]} waist_roll_delta={best3[2]} waist_pitch_delta={best3[3]} "
          f"|qacc_free|={best3[0]:.3f} fz_R={best3[5]:.2f}")
    print(f"  qacc_free 분해(lin xyz, ang xyz): {best3[6]}")

    np.save("best_one_leg_qpos.npy", best3[7])
    print("저장: best_one_leg_qpos.npy")


if __name__ == "__main__":
    main()
