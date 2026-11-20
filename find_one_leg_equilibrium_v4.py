"""find_one_leg_equilibrium_v4.py
----------------------------------
사용자 피드백(뷰어로 직접 보고): "드는 다리(왼다리)를 옆으로
벌리며(hip_abduct_L) 들어야 한다" - v3는 무릎(knee_L)+고관절
굽힘(hip_L)만으로 들었는데, 옆으로 벌리는 동작(hip_abduct_L)을
추가한다. leg_L은 y<0 쪽에 붙어있고 hip_abduct_L 축은 x축 -
회전행렬 Rx(θ)를 (0,0,-1)(아래로 뻗은 다리)에 적용하면 θ>0일 때
y가 +방향(중앙쪽, 즉 모음/adduction)으로 움직인다 - 따라서
밖으로 벌리려면(abduction) **음수** hip_abduct_L이 필요하다.

v3의 5개 파라미터(theta1, waist_roll, waist_pitch, knee_L, hip_L)에
hip_abduct_L을 6번째로 추가해 가우스-뉴턴을 다시 돈다.
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

    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    target_foot_pos = data0.xpos[foot_R_id].copy()
    target_foot_mat = data0.xmat[foot_R_id].copy().reshape(3, 3)

    # 왼다리 들기 파라미터를 고정값으로 넉넉하게 잡아뒀으니(서있는
    # 골반 기준 38cm 클리어런스 확인됨) 탐색 중엔 다시 충돌을 꺼서
    # 오른다리/상체 균형(3개 변수)만 깨끗하게 푼다 - 끝나고 충돌 켠
    # 채로 실제 클리어런스를 검증한다.
    foot_L_geom = model.body_geomadr[foot_L_id]
    shin_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shin_L")
    shin_L_geom = model.body_geomadr[shin_L_id]
    model.geom_contype[foot_L_geom] = 0
    model.geom_conaffinity[foot_L_geom] = 0
    model.geom_contype[shin_L_geom] = 0
    model.geom_conaffinity[shin_L_geom] = 0

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
    q_ankle_pitch_L = jq("ankle_pitch_L")
    q_ankle_roll_L = jq("ankle_roll_L")
    q_waist_roll = jq("waist_roll")
    q_waist_pitch = jq("waist_pitch")

    theta2 = qpos_eq0[q_hip_R]
    theta3 = qpos_eq0[q_knee_R]
    theta4 = qpos_eq0[q_ankle_roll_R]
    theta5 = qpos_eq0[q_ankle_pitch_R]
    waist_roll_0 = qpos_eq0[q_waist_roll]
    waist_pitch_0 = qpos_eq0[q_waist_pitch]

    # 왼다리 들기 파라미터는 고정 (서있는 골반 기준으로 ~38cm 클리어런스
    # 확인된 넉넉한 값) - 균형 잡느라 골반이 내려가도 여유가 있도록
    ABDUCT_L_DEG = -25.0
    KNEE_L_DEG = 45.0
    HIP_L_DEG = 15.0

    def build_qpos(p):
        theta1_deg, wroll_deg, wpitch_deg = p
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
        qpos[q_knee_L] = math.radians(KNEE_L_DEG)
        qpos[q_hip_L] = math.radians(HIP_L_DEG)
        qpos[q_abduct_L] = math.radians(ABDUCT_L_DEG)
        qpos[q_ankle_pitch_L] = 0.0
        qpos[q_ankle_roll_L] = 0.0
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

    print("=== 가우스-뉴턴(백트래킹) - 왼다리 고정, theta1/waist 3개만 최적화 ===")
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
    fz_L = foot_fz(model, data_final, foot_L_id)
    fz_R = foot_fz(model, data_final, foot_R_id)
    total_weight_N = sum(model.body_mass) * 9.81
    print()
    print(f"최종: p={p}")
    print(f"|qacc_free|={np.linalg.norm(f_final):.5f}")
    print(f"fz_L={fz_L:.2f} fz_R={fz_R:.2f} (전체 몸무게={total_weight_N:.2f})")

    np.save("best_one_leg_qpos_v4.npy", qpos_final)
    print("저장: best_one_leg_qpos_v4.npy")

    print()
    print("=== 충돌을 다시 켠 채로 실제 클리어런스 검증 ===")
    model2 = common.load_model()  # 충돌 설정이 기본값(켜짐)인 새 모델
    data2 = mujoco.MjData(model2)
    data2.qpos[:] = qpos_final
    mujoco.mj_forward(model2, data2)
    fz_L2 = foot_fz(model2, data2, foot_L_id)
    print(f"충돌 켠 상태 fz_L={fz_L2:.2f} (0이어야 진짜 한 발 지지)")
    print(f"foot_L body z={data2.xpos[foot_L_id][2]:.4f}, foot_L geom z={data2.geom_xpos[foot_L_geom][2]:.4f}")
    for i in range(data2.ncon):
        c = data2.contact[i]
        b1 = model2.geom_bodyid[c.geom1]
        b2 = model2.geom_bodyid[c.geom2]
        n1 = mujoco.mj_id2name(model2, mujoco.mjtObj.mjOBJ_BODY, b1)
        n2 = mujoco.mj_id2name(model2, mujoco.mjtObj.mjOBJ_BODY, b2)
        print(f"  contact: {n1} - {n2} dist={c.dist:.4f}")


if __name__ == "__main__":
    main()
