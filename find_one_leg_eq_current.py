"""find_one_leg_eq_current.py
------------------------------
find_one_leg_equilibrium_v2.py와 같은 역기구학 방법이지만, 오늘 바뀐
로봇 구조(정강이-바닥 팬텀 접촉 수정, 골반 폭 0.40, 발 오프셋
0.28/0.16, 다리 kp/kv x3) 위에서 다시 찾는다. best_one_leg_qpos_v5.npy는
이 구조 변경 전에 만들어진 것이라 그대로 쓰면 안 맞는다.

스탠스 발(foot_L, 파란색)은 그대로, 스윙 다리(hip_R/knee_R)는 오늘
lift_cp_*.py에서 쓰던 목표값(knee_R=30deg, hip_R=20deg)으로 고정해두고,
theta1(hip_abduct_L 리언) + waist_roll + waist_pitch 를 스윕해서 몸통
자유관절 가속도(=CoM이 파란발 위로 오는 정도)가 가장 작은 지점을 찾는다.
"""
import math

import numpy as np
import mujoco

import common
from pelvis_width_test import widen_pelvis
from foot_offset_test import set_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv
from find_one_leg_equilibrium_v2 import Rx, Ry, mat2quat, foot_fz

SWING_KNEE_DEG = 30.0
SWING_HIP_DEG = 20.0


def solve_torso_for_lean_general(theta1, theta2, theta3, theta4, theta5,
                                  target_foot_pos, target_foot_mat, pelvis_offset):
    """find_one_leg_equilibrium_v2.solve_torso_for_lean과 같은 역기구학이지만,
    원래 버전은 골반 오프셋 [0,0.25,-0.7]을 상수로 박아놔서 widen_pelvis()로
    폭을 바꾸면(0.40) 안 맞는다 - pelvis_offset을 실제 모델에서 읽어 넘긴다."""
    R_b = Rx(theta1) @ Ry(theta2)
    R_c = R_b @ Ry(theta3)
    R_e = R_c @ Rx(theta4) @ Ry(theta5)
    p_chain = pelvis_offset + R_b @ np.array([0.0, 0.0, -0.6]) \
        + R_c @ np.array([0.0, 0.0, -0.6])
    R0 = target_foot_mat @ R_e.T
    P0 = target_foot_pos - R0 @ p_chain
    return P0, R0


def build_model():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.28, 0.16, 0.0)
    scale_kp_kv(model, LEG_ACTUATORS, 3.0)
    model.opt.timestep = 0.0005
    return model


def main():
    model = build_model()
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)

    data0 = mujoco.MjData(model)
    data0.qpos[:] = qpos_eq0
    data0.qvel[:] = 0.0
    mujoco.mj_forward(model, data0)

    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    target_foot_pos = data0.xpos[foot_L_id].copy()
    target_foot_mat = data0.xmat[foot_L_id].copy().reshape(3, 3)

    # 빨간발(swing)은 바닥과 충돌 비활성화 - 순수하게 "파란발 하나로만
    # 지지"하는 기하학적 조건으로 CoM 위치만 본다
    foot_R_geom = model.body_geomadr[foot_R_id]
    model.geom_contype[foot_R_geom] = 0
    model.geom_conaffinity[foot_R_geom] = 0

    leg_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_L")
    pelvis_offset = model.body_pos[leg_L_id].copy()

    def jq(name):
        return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]

    q_abduct_L = jq("hip_abduct_L")
    q_hip_L, q_knee_L = jq("hip_L"), jq("knee_L")
    q_ankle_roll_L, q_ankle_pitch_L = jq("ankle_roll_L"), jq("ankle_pitch_L")
    q_abduct_R, q_hip_R, q_knee_R = jq("hip_abduct_R"), jq("hip_R"), jq("knee_R")
    q_ankle_roll_R, q_ankle_pitch_R = jq("ankle_roll_R"), jq("ankle_pitch_R")
    q_waist_roll, q_waist_pitch = jq("waist_roll"), jq("waist_pitch")

    theta2 = qpos_eq0[q_hip_L]
    theta3 = qpos_eq0[q_knee_L]
    theta4 = qpos_eq0[q_ankle_roll_L]
    theta5 = qpos_eq0[q_ankle_pitch_L]
    waist_roll_0 = qpos_eq0[q_waist_roll]
    waist_pitch_0 = qpos_eq0[q_waist_pitch]

    total_weight_N = sum(model.body_mass) * 9.81
    print(f"전체 몸무게(N)={total_weight_N:.2f}")

    P0_check, R0_check = solve_torso_for_lean_general(0.0, theta2, theta3, theta4, theta5,
                                                        target_foot_pos, target_foot_mat, pelvis_offset)
    print("sanity check (theta1=0): P0", P0_check, "vs 실제", qpos_eq0[0:3])

    def build_qpos(theta1_deg, waist_roll_delta_deg, waist_pitch_delta_deg):
        theta1 = math.radians(theta1_deg)
        P0, R0 = solve_torso_for_lean_general(theta1, theta2, theta3, theta4, theta5,
                                               target_foot_pos, target_foot_mat, pelvis_offset)
        quat0 = mat2quat(R0)
        qpos = qpos_eq0.copy()
        qpos[0:3] = P0
        qpos[3:7] = quat0
        qpos[q_abduct_L] = theta1
        qpos[q_waist_roll] = waist_roll_0 + math.radians(waist_roll_delta_deg)
        qpos[q_waist_pitch] = waist_pitch_0 + math.radians(waist_pitch_delta_deg)
        qpos[q_knee_R] = math.radians(SWING_KNEE_DEG)
        qpos[q_hip_R] = math.radians(SWING_HIP_DEG)
        qpos[q_abduct_R] = 0.0
        qpos[q_ankle_pitch_R] = 0.0
        qpos[q_ankle_roll_R] = 0.0
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
        return qacc_free, fz_L

    print("=== 1차: theta1 스윕 ===")
    best = None
    for theta1_deg in np.arange(-30, -5, 1.0):
        qpos = build_qpos(theta1_deg, 0.0, 0.0)
        qacc_free, fz_L = eval_qpos(qpos)
        acc_norm = np.linalg.norm(qacc_free)
        if best is None or acc_norm < best[0]:
            best = (acc_norm, theta1_deg, 0.0, 0.0, fz_L)
    print(f"1차 최적: theta1={best[1]} |qacc_free|={best[0]:.3f} fz_L={best[4]:.2f}")

    print("=== 2차: theta1 + waist_roll 2D 스윕 ===")
    theta1_c = best[1]
    best2 = None
    for theta1_deg in np.arange(theta1_c - 6, theta1_c + 6, 1.0):
        for wroll in np.arange(-20, 21, 2.0):
            qpos = build_qpos(theta1_deg, wroll, 0.0)
            qacc_free, fz_L = eval_qpos(qpos)
            acc_norm = np.linalg.norm(qacc_free)
            if best2 is None or acc_norm < best2[0]:
                best2 = (acc_norm, theta1_deg, wroll, 0.0, fz_L)
    print(f"2차 최적: theta1={best2[1]} waist_roll_delta={best2[2]} "
          f"|qacc_free|={best2[0]:.3f} fz_L={best2[4]:.2f}")

    print("=== 3차: theta1 + waist_roll + waist_pitch 3D 스윕 ===")
    theta1_c, wroll_c = best2[1], best2[2]
    best3 = None
    for theta1_deg in np.arange(theta1_c - 4, theta1_c + 4, 1.0):
        for wroll in np.arange(wroll_c - 6, wroll_c + 6, 2.0):
            for wpitch in np.arange(-25, 26, 2.0):
                qpos = build_qpos(theta1_deg, wroll, wpitch)
                qacc_free, fz_L = eval_qpos(qpos)
                acc_norm = np.linalg.norm(qacc_free)
                if best3 is None or acc_norm < best3[0]:
                    best3 = (acc_norm, theta1_deg, wroll, wpitch, fz_L, qpos.copy())
    print(f"3차 최적: theta1={best3[1]} waist_roll_delta={best3[2]} waist_pitch_delta={best3[3]} "
          f"|qacc_free|={best3[0]:.3f} fz_L={best3[4]:.2f} (목표={total_weight_N:.1f})")

    print("=== 4차: 미세 조정(0.5도 간격) ===")
    theta1_c, wroll_c, wpitch_c = best3[1], best3[2], best3[3]
    best4 = None
    for theta1_deg in np.arange(theta1_c - 3, theta1_c + 3, 0.5):
        for wroll in np.arange(wroll_c - 4, wroll_c + 4, 0.5):
            for wpitch in np.arange(wpitch_c - 4, wpitch_c + 4, 0.5):
                qpos = build_qpos(theta1_deg, wroll, wpitch)
                qacc_free, fz_L = eval_qpos(qpos)
                acc_norm = np.linalg.norm(qacc_free)
                if best4 is None or acc_norm < best4[0]:
                    best4 = (acc_norm, theta1_deg, wroll, wpitch, fz_L, qpos.copy())
    print(f"4차 최적: theta1={best4[1]} waist_roll_delta={best4[2]} waist_pitch_delta={best4[3]} "
          f"|qacc_free|={best4[0]:.3f} fz_L={best4[4]:.2f} (목표={total_weight_N:.1f})")

    np.save("best_one_leg_qpos_current.npy", best4[5])
    print("저장: best_one_leg_qpos_current.npy")


if __name__ == "__main__":
    main()
