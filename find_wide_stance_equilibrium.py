"""find_wide_stance_equilibrium.py
-----------------------------------
"오른발을 옆으로 살짝 벌려서 딛는다"는 - 발이 공중에 뜨는 게
아니라 **새 위치에 다시 땅을 짚는** 거라서, 목표점이 한 발 지지가
아니라 **더 넓은 두 발 서있기**다. 훨씬 쉬운 문제 - 양발 다 접촉이
있는 안정적인 평형점이면 된다.

오른다리(hip_abduct_R)를 바깥으로 theta_R만큼 벌리고, 발을 평평하게
유지하도록 ankle_roll_R로 자체 상쇄(-theta_R, solve_torso_for_lean과
같은 수학) - 이러면 오른발이 옆으로 L*sin(theta_R)만큼 이동하고
약간 높이도 바뀐다. 왼다리는 기존처럼 스탠스 기준(solve_torso_for_lean)
으로 쓰고, 왼다리 쪽 작은 리언(theta_L)을 가우스-뉴턴으로 찾아서
전체 균형(qacc_free=0)을 맞춘다 - 양발 충돌 다 켠 채로(진짜 두
발 서있기).
"""
import math

import numpy as np
import mujoco

import common
from find_one_leg_equilibrium_v2 import Rx, Ry, mat2quat, solve_torso_for_lean, foot_fz
from pelvis_width_test import widen_pelvis
from foot_offset_test import set_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv


def build_best_model():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.28, 0.16, 0.0)
    scale_kp_kv(model, LEG_ACTUATORS, 3.0)
    model.opt.timestep = 0.0005
    return model


def main():
    model = build_best_model()
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)

    data0 = mujoco.MjData(model)
    data0.qpos[:] = qpos_eq0
    data0.qvel[:] = 0.0
    mujoco.mj_forward(model, data0)

    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    target_foot_pos = data0.xpos[foot_L_id].copy()
    target_foot_mat = data0.xmat[foot_L_id].copy().reshape(3, 3)

    def jq(name):
        return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]

    q_abduct_L = jq("hip_abduct_L")
    q_hip_L, q_knee_L = jq("hip_L"), jq("knee_L")
    q_ankle_roll_L, q_ankle_pitch_L = jq("ankle_roll_L"), jq("ankle_pitch_L")
    q_abduct_R = jq("hip_abduct_R")
    q_hip_R, q_knee_R = jq("hip_R"), jq("knee_R")
    q_ankle_roll_R, q_ankle_pitch_R = jq("ankle_roll_R"), jq("ankle_pitch_R")
    q_waist_roll, q_waist_pitch = jq("waist_roll"), jq("waist_pitch")

    theta2 = qpos_eq0[q_hip_L]
    theta3 = qpos_eq0[q_knee_L]
    theta4 = qpos_eq0[q_ankle_roll_L]
    theta5 = qpos_eq0[q_ankle_pitch_L]
    waist_roll_0 = qpos_eq0[q_waist_roll]
    waist_pitch_0 = qpos_eq0[q_waist_pitch]

    THETA_R_DEG = 6.0  # 오른발을 옆으로 벌리는 각도("살짝") - 15도는
    # 실측 0.46m 이동으로 너무 커서(보정도 과해짐) 6도로 축소

    # 오른 무릎을 더 굽히면 다리가 짧아져서 발이 오히려 더 올라간다
    # (방향이 반대) - 오른발을 다시 땅에 닿게 하려면 왼(지지)다리
    # 무릎을 살짝 굽혀서 골반 자체를 낮춰야 한다(골반이 내려가면
    # 오른발도 같이 내려감). knee_L_delta를 자유변수로.
    def build_qpos(p):
        theta1_deg, wroll_deg, wpitch_deg, knee_L_delta_deg = p
        theta1 = math.radians(theta1_deg)
        theta3_adj = theta3 + math.radians(knee_L_delta_deg)
        P0, R0 = solve_torso_for_lean(theta1, theta2, theta3_adj, theta4, theta5,
                                       target_foot_pos, target_foot_mat)
        quat0 = mat2quat(R0)
        qpos = qpos_eq0.copy()
        qpos[0:3] = P0
        qpos[3:7] = quat0
        qpos[q_abduct_L] = theta1
        qpos[q_knee_L] = theta3_adj
        qpos[q_waist_roll] = waist_roll_0 + math.radians(wroll_deg)
        qpos[q_waist_pitch] = waist_pitch_0 + math.radians(wpitch_deg)
        # 오른다리: 바깥으로 벌리고 발목으로 자체 상쇄(평평하게 유지).
        # 높이 보정은 오른 무릎이 아니라 왼(지지) 무릎의 knee_L_delta로.
        qpos[q_abduct_R] = math.radians(THETA_R_DEG)
        qpos[q_ankle_roll_R] = qpos_eq0[q_ankle_roll_R] - math.radians(THETA_R_DEG)
        qpos[q_hip_R] = qpos_eq0[q_hip_R]
        qpos[q_knee_R] = qpos_eq0[q_knee_R]
        qpos[q_ankle_pitch_R] = qpos_eq0[q_ankle_pitch_R]
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

    # 오른발이 실제로 땅에 어떻게 닿는지 먼저 확인(기구학만, 가속도 계산 전)
    qpos_check = build_qpos([0.0, 0.0, 0.0, 0.0])
    data_check = mujoco.MjData(model)
    data_check.qpos[:] = qpos_check
    mujoco.mj_forward(model, data_check)
    print(f"theta_R={THETA_R_DEG}도일 때 오른발 위치: {data_check.xpos[foot_R_id]}")
    print(f"기존 오른발 위치(참고): {data0.xpos[foot_R_id]}")
    print(f"오른발 y이동(벌어진 폭): {data_check.xpos[foot_R_id][1] - data0.xpos[foot_R_id][1]:.3f}m")

    p = np.array([0.0, 0.0, 0.0, 2.0])  # knee_L_delta 초기값: 살짝 굽혀서 골반 낮추기
    h = np.array([0.3, 0.3, 0.3, 0.3])

    print()
    print("=== 가우스-뉴턴(백트래킹) - 왼다리 리언+무릎(골반 높이)로 균형 찾기 ===")
    f0, _, _ = qacc_free_of(p)
    norm0 = np.linalg.norm(f0)
    print(f"시작: |qacc_free|={norm0:.4f}")
    for it in range(40):
        J = np.zeros((6, 4))
        for j in range(4):
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
    print(f"fz_L={fz_L:.2f} fz_R={fz_R:.2f} (전체 몸무게={total_weight_N:.2f}, 둘 다 0보다 커야 진짜 두 발 서있기)")

    np.save("best_wide_stance_qpos.npy", qpos_final)
    print("저장: best_wide_stance_qpos.npy")


if __name__ == "__main__":
    main()
