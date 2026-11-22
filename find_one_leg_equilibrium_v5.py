"""find_one_leg_equilibrium_v5.py
----------------------------------
v4는 스윙 다리(로봇 오른쪽) 들기 값을 "클리어런스 안전 마진"
기준으로 크게 고정했다(abduct=-25, knee=45, hip=15) - 비자연스럽고
TV-LQR이 그 큰 무릎 동작을 못 따라갔다.

사용자 제안: 고관절 외전(abduction)만으로도 꽤 들린다(-20도만으로
땅에서 7cm) - 무릎/고관절 pitch는 그걸 보조하는 정도로 살짝만.
이번엔 abduct=-20, knee=10, hip=5(클리어런스 실측 7.6cm)로 훨씬
자연스러운 조합을 고정하고 균형(theta1/waist)을 다시 푼다.
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
    shin_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shin_R")
    shin_R_geom = model.body_geomadr[shin_R_id]
    model.geom_contype[foot_R_geom] = 0
    model.geom_conaffinity[foot_R_geom] = 0
    model.geom_contype[shin_R_geom] = 0
    model.geom_conaffinity[shin_R_geom] = 0

    def jq(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid]

    q_abduct_L = jq("hip_abduct_L")
    q_hip_L = jq("hip_L")
    q_knee_L = jq("knee_L")
    q_ankle_roll_L = jq("ankle_roll_L")
    q_ankle_pitch_L = jq("ankle_pitch_L")
    q_abduct_R = jq("hip_abduct_R")
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

    # 1차 시도(abduct=-20,knee=10,hip=5, 클리어런스 7.6cm)는 실패 -
    # theta1=-25도일 때 골반이 17cm나 내려가서 그 클리어런스로는
    # 턱없이 부족했다. 골반 하강(이론상 ~17-20cm)을 감당할 만큼
    # 여유를 주되(실측 클리어런스 ~23cm), v4(45/15, 38cm)보다는
    # 덜 과장된 조합으로.
    ABDUCT_R_DEG = -25.0
    KNEE_R_DEG = 35.0
    HIP_R_DEG = 10.0

    def build_qpos(p):
        theta1_deg, wroll_deg, wpitch_deg = p
        theta1 = math.radians(theta1_deg)
        P0, R0 = solve_torso_for_lean(theta1, theta2, theta3, theta4, theta5,
                                       target_foot_pos, target_foot_mat)
        quat0 = mat2quat(R0)
        qpos = qpos_eq0.copy()
        qpos[0:3] = P0
        qpos[3:7] = quat0
        qpos[q_abduct_L] = theta1
        qpos[q_waist_roll] = waist_roll_0 + math.radians(wroll_deg)
        qpos[q_waist_pitch] = waist_pitch_0 + math.radians(wpitch_deg)
        qpos[q_knee_R] = math.radians(KNEE_R_DEG)
        qpos[q_hip_R] = math.radians(HIP_R_DEG)
        qpos[q_abduct_R] = math.radians(ABDUCT_R_DEG)
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

    p = np.array([-9.0, 1.0, 0.0])
    h = np.array([0.3, 0.3, 0.3])

    print("=== 가우스-뉴턴(백트래킹) - 자연스러운 스윙 조합 고정, theta1/waist 최적화 ===")
    f0, _, _ = qacc_free_of(p)
    norm0 = np.linalg.norm(f0)
    print(f"시작: |qacc_free|={norm0:.4f}")
    for it in range(30):
        J = np.zeros((6, 3))
        for j in range(3):
            pp = p.copy()
            pp[j] += h[j]
            fp, _, _ = qacc_free_of(pp)
            J[:, j] = (fp - f0) / h[j]
        dp, *_ = np.linalg.lstsq(J, -f0, rcond=None)
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
    print(f"fz_swing={fz_L:.2f} fz_stance={fz_R:.2f} (전체 몸무게={total_weight_N:.2f})")

    np.save("best_one_leg_qpos_v5.npy", qpos_final)
    print("저장: best_one_leg_qpos_v5.npy")

    print()
    print("=== 충돌을 다시 켠 채로 실제 클리어런스 검증 ===")
    model2 = common.load_model()
    data2 = mujoco.MjData(model2)
    data2.qpos[:] = qpos_final
    mujoco.mj_forward(model2, data2)
    fz_swing2 = foot_fz(model2, data2, foot_R_id)
    print(f"충돌 켠 상태 fz_swing={fz_swing2:.2f} (0이어야 진짜 한 발 지지)")
    print(f"foot_R body z={data2.xpos[foot_R_id][2]:.4f}, foot_R geom z={data2.geom_xpos[foot_R_geom][2]:.4f}")
    for i in range(data2.ncon):
        c = data2.contact[i]
        b1 = model2.geom_bodyid[c.geom1]
        b2 = model2.geom_bodyid[c.geom2]
        n1 = mujoco.mj_id2name(model2, mujoco.mjtObj.mjOBJ_BODY, b1)
        n2 = mujoco.mj_id2name(model2, mujoco.mjtObj.mjOBJ_BODY, b2)
        print(f"  contact: {n1} - {n2} dist={c.dist:.4f}")


if __name__ == "__main__":
    main()
