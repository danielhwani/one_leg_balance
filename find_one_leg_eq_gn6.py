"""find_one_leg_eq_gn6.py
--------------------------
find_one_leg_eq_gn.py는 스윙 다리 포즈(abduct_R=-25/knee_R=35/hip_R=10)를
v5(원래/좁은 구조에서 찾은 값)로 고정해놓고 theta1/waist_roll/waist_pitch
3개만 풀었다 - 지금(넓은 골반) 구조에 맞게 재최적화된 적이 없다.
이번엔 6개 변수(theta1, waist_roll, waist_pitch, abduct_R, knee_R, hip_R)를
모두 풀어서, 이 구조 전용으로 더 낮은 |qacc_free| 평형점을 찾는다.
"""
import math

import numpy as np
import mujoco

import common
from pelvis_width_test import widen_pelvis
from foot_offset_test import set_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv
from find_one_leg_equilibrium_v2 import mat2quat, foot_fz
from find_one_leg_eq_current import solve_torso_for_lean_general


def build_current_model():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.28, 0.16, 0.0)
    scale_kp_kv(model, LEG_ACTUATORS, 1.0)
    model.opt.timestep = 0.0005
    return model


def main():
    model = build_current_model()
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    data0 = mujoco.MjData(model)
    data0.qpos[:] = qpos_eq0
    mujoco.mj_forward(model, data0)

    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    shin_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shin_R")
    target_foot_pos = data0.xpos[foot_L_id].copy()
    target_foot_mat = data0.xmat[foot_L_id].copy().reshape(3, 3)

    foot_R_geom = model.body_geomadr[foot_R_id]
    shin_R_geom = model.body_geomadr[shin_R_id]
    model.geom_contype[foot_R_geom] = 0
    model.geom_conaffinity[foot_R_geom] = 0
    model.geom_contype[shin_R_geom] = 0
    model.geom_conaffinity[shin_R_geom] = 0

    leg_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_L")
    pelvis_offset = model.body_pos[leg_L_id].copy()

    def jq(name):
        return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]

    q_abduct_L = jq("hip_abduct_L")
    q_hip_L, q_knee_L = jq("hip_L"), jq("knee_L")
    q_ankle_roll_L, q_ankle_pitch_L = jq("ankle_roll_L"), jq("ankle_pitch_L")
    q_abduct_R, q_hip_R, q_knee_R = jq("hip_abduct_R"), jq("hip_R"), jq("knee_R")
    q_ankle_pitch_R, q_ankle_roll_R = jq("ankle_pitch_R"), jq("ankle_roll_R")
    q_waist_roll, q_waist_pitch = jq("waist_roll"), jq("waist_pitch")

    theta2 = qpos_eq0[q_hip_L]
    theta3 = qpos_eq0[q_knee_L]
    theta4 = qpos_eq0[q_ankle_roll_L]
    theta5 = qpos_eq0[q_ankle_pitch_L]
    waist_roll_0 = qpos_eq0[q_waist_roll]
    waist_pitch_0 = qpos_eq0[q_waist_pitch]

    def build_qpos(p):
        theta1_deg, wroll_deg, wpitch_deg, abduct_r_deg, knee_r_deg, hip_r_deg = p
        theta1 = math.radians(theta1_deg)
        P0, R0 = solve_torso_for_lean_general(theta1, theta2, theta3, theta4, theta5,
                                               target_foot_pos, target_foot_mat, pelvis_offset)
        quat0 = mat2quat(R0)
        qpos = qpos_eq0.copy()
        qpos[0:3] = P0
        qpos[3:7] = quat0
        qpos[q_abduct_L] = theta1
        qpos[q_waist_roll] = waist_roll_0 + math.radians(wroll_deg)
        qpos[q_waist_pitch] = waist_pitch_0 + math.radians(wpitch_deg)
        qpos[q_knee_R] = math.radians(knee_r_deg)
        qpos[q_hip_R] = math.radians(hip_r_deg)
        qpos[q_abduct_R] = math.radians(abduct_r_deg)
        qpos[q_ankle_pitch_R] = 0.0
        qpos[q_ankle_roll_R] = 0.0
        return qpos

    def qacc_free_of(p):
        qpos = build_qpos(p)
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        for i in range(model.nu):
            jid = model.actuator_trnid[i, 0]
            qadr = model.jnt_qposadr[jid]
            data.ctrl[i] = qpos[qadr]
        mujoco.mj_forward(model, data)
        return data.qacc[0:6].copy(), qpos, data

    p = np.array([-25.0, 6.0, 1.5, -25.0, 35.0, 10.0])  # gn.py 결과 + v5 스윙포즈로 시작
    h = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    f0, _, _ = qacc_free_of(p)
    norm0 = np.linalg.norm(f0)
    print(f"시작: |qacc_free|={norm0:.4f} p={np.round(p,2)}")
    for it in range(60):
        J = np.zeros((6, 6))
        for j in range(6):
            pp = p.copy()
            pp[j] += h[j]
            fp, _, _ = qacc_free_of(pp)
            J[:, j] = (fp - f0) / h[j]
        dp, *_ = np.linalg.lstsq(J, -f0, rcond=None)
        scale = 1.0
        p_try = p
        for _ in range(8):
            p_try = p + scale * dp
            f_try, _, _ = qacc_free_of(p_try)
            if np.linalg.norm(f_try) < norm0:
                break
            scale *= 0.5
        p = p_try
        f0 = f_try
        norm0 = np.linalg.norm(f0)
        if it % 5 == 0:
            print(f"iter {it}: |qacc_free|={norm0:9.5f} scale={scale:.3f} p={np.round(p, 2)}")
        if norm0 < 0.05:
            print("수렴!")
            break
    f_final, qpos_final, data_final = qacc_free_of(p)
    fz_L = foot_fz(model, data_final, foot_L_id)
    total_weight_N = sum(model.body_mass) * 9.81
    print(f"\n최종: p={np.round(p,3)}")
    print(f"|qacc_free|={np.linalg.norm(f_final):.5f}")
    print(f"fz_stance={fz_L:.2f} (목표={total_weight_N:.2f})")
    np.save("best_one_leg_qpos_current_gn6.npy", qpos_final)
    print("저장: best_one_leg_qpos_current_gn6.npy")


if __name__ == "__main__":
    main()
