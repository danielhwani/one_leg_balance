"""find_one_leg_equilibrium_v3.py
----------------------------------
v2의 그리드 탐색으로 |qacc_free|를 8.6 -> 5.6 -> 3.38까지 줄였지만
grid라서 느리고 한계가 있다. 여기선 v2의 최적 후보(theta1=-12,
waist_roll_delta=-10, waist_pitch_delta=3)를 시작점으로, 수치
자코비안 + 가우스-뉴턴(least squares)으로 5개 자유 파라미터
(theta1, waist_roll_delta, waist_pitch_delta, knee_R, hip_R)를
조정해 6개 성분(lin xyz, ang xyz)의 qacc_free를 직접 0으로 미는
작업을 반복한다.
"""
import math

import numpy as np
import mujoco

import common
from find_one_leg_equilibrium_v2 import Rx, Ry, mat2quat, solve_torso_for_lean, foot_fz


def main():
    model = common.load_model()
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)

    data0 = mujoco.MjData(model)
    data0.qpos[:] = qpos_eq0
    data0.qvel[:] = 0.0
    mujoco.mj_forward(model, data0)

    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    target_foot_pos = data0.xpos[foot_L_id].copy()
    target_foot_mat = data0.xmat[foot_L_id].copy().reshape(3, 3)

    foot_R_geom = model.body_geomadr[foot_R_id]
    model.geom_contype[foot_R_geom] = 0
    model.geom_conaffinity[foot_R_geom] = 0

    def jq(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid]

    q_abduct_R = jq("hip_abduct_L")
    q_hip_L = jq("hip_L")
    q_knee_L = jq("knee_L")
    q_ankle_roll_L = jq("ankle_roll_L")
    q_ankle_pitch_L = jq("ankle_pitch_L")
    q_hip_R = jq("hip_R")
    q_knee_R = jq("knee_R")
    q_ankle_pitch_R = jq("ankle_pitch_R")
    q_ankle_roll_R = jq("ankle_roll_R")
    q_waist_roll = jq("waist_roll")
    q_waist_pitch = jq("waist_pitch")

    theta2 = qpos_eq0[q_hip_L]
    theta3 = qpos_eq0[q_knee_L]
    theta4 = qpos_eq0[q_ankle_roll_L]
    theta5 = qpos_eq0[q_ankle_pitch_L]
    waist_roll_0 = qpos_eq0[q_waist_roll]
    waist_pitch_0 = qpos_eq0[q_waist_pitch]

    def build_qpos(p):
        theta1_deg, wroll_deg, wpitch_deg, kneeL_deg, hipL_deg = p
        theta1 = math.radians(theta1_deg)
        P0, R0 = solve_torso_for_lean(theta1, theta2, theta3, theta4, theta5,
                                       target_foot_pos, target_foot_mat)
        quat0 = mat2quat(R0)
        qpos = qpos_eq0.copy()
        qpos[0:3] = P0
        qpos[3:7] = quat0
        qpos[q_abduct_R] = theta1
        qpos[q_waist_roll] = waist_roll_0 + math.radians(wroll_deg)
        qpos[q_waist_pitch] = waist_pitch_0 + math.radians(wpitch_deg)
        qpos[q_knee_R] = math.radians(kneeL_deg)
        qpos[q_hip_R] = math.radians(hipL_deg)
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

    p = np.array([-15.84218057, 2.4604867, -0.28658927, 25.1757404, 4.60143382])
    h = np.array([0.3, 0.3, 0.3, 0.3, 0.3])

    print("=== 가우스-뉴턴(백트래킹 포함)으로 qacc_free -> 0 반복 ===")
    f0, _, _ = qacc_free_of(p)
    norm0 = np.linalg.norm(f0)
    for it in range(25):
        J = np.zeros((6, 5))
        for j in range(5):
            pp = p.copy()
            pp[j] += h[j]
            fp, _, _ = qacc_free_of(pp)
            J[:, j] = (fp - f0) / h[j]
        dp, *_ = np.linalg.lstsq(J, -f0, rcond=None)
        # 백트래킹: 전체 스텝이 안 좋아지면 절반씩 줄여서 받아들임
        scale = 1.0
        for _ in range(6):
            p_try = p + scale * dp
            f_try, _, _ = qacc_free_of(p_try)
            if np.linalg.norm(f_try) < norm0:
                break
            scale *= 0.5
        p = p_try
        f0 = f_try
        norm0 = np.linalg.norm(f0)
        print(f"iter {it}: |qacc_free|={norm0:9.5f} scale={scale:.3f} p={np.round(p, 3)}")
        if norm0 < 0.05:
            print("수렴!")
            break

    f_final, qpos_final, data_final = qacc_free_of(p)
    fz_L = foot_fz(model, data_final, foot_R_id)
    fz_R = foot_fz(model, data_final, foot_L_id)
    total_weight_N = sum(model.body_mass) * 9.81
    print()
    print(f"최종: p={p}")
    print(f"|qacc_free|={np.linalg.norm(f_final):.5f}")
    print(f"qacc_free 분해: {f_final}")
    print(f"fz_L={fz_L:.2f} fz_R={fz_R:.2f} (전체 몸무게={total_weight_N:.2f})")

    np.save("best_one_leg_qpos.npy", qpos_final)
    print("저장: best_one_leg_qpos.npy")


if __name__ == "__main__":
    main()
