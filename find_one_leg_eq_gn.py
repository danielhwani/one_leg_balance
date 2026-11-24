"""find_one_leg_eq_gn.py
-------------------------
find_one_leg_eq_current.py의 거친 그리드 탐색(|qacc_free|~2.84)은
find_one_leg_equilibrium_v5.py가 쓴 가우스-뉴턴 방식(|qacc_free|<0.05,
50배 더 정밀)보다 훨씬 부정확했다. 또 v5는 스윙 다리를 "무릎만 굽히는"
자세(knee=30,hip=20,abduct=0)가 아니라 "고관절 외전으로 벌려서 지면
클리어런스를 만드는" 자연스러운 조합(abduct=-25,knee=35,hip=10)을
썼다 - 오늘 lift_straight_up_test.py(순수 수직 들기)가
lift_and_swing_test.py(벌리면서 들기)보다 더 나빴던 것과 일치한다.

이 스크립트는 v5와 같은 가우스-뉴턴 + 자연스러운 스윙 조합을, 좁은
설정과 현재(넓은) 설정 둘 다에 적용해서 정밀한 평형점을 다시 찾는다.
"""
import math

import numpy as np
import mujoco

import common
from pelvis_width_test import widen_pelvis
from foot_offset_test import set_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv
from find_one_leg_equilibrium_v2 import Rx, Ry, mat2quat, foot_fz
from find_one_leg_eq_current import solve_torso_for_lean_general

ABDUCT_R_DEG = -25.0
KNEE_R_DEG = 35.0
HIP_R_DEG = 10.0


def build_narrow_model():
    return common.load_model()


def build_current_model():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.28, 0.16, 0.0)
    scale_kp_kv(model, LEG_ACTUATORS, 3.0)
    model.opt.timestep = 0.0005
    return model


def solve_equilibrium(model, out_path, p0=(-9.0, 1.0, 0.0)):
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
        theta1_deg, wroll_deg, wpitch_deg = p
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

    p = np.array(p0)
    h = np.array([0.3, 0.3, 0.3])
    f0, _, _ = qacc_free_of(p)
    norm0 = np.linalg.norm(f0)
    print(f"  시작: |qacc_free|={norm0:.4f}")
    for it in range(30):
        J = np.zeros((6, 3))
        for j in range(3):
            pp = p.copy()
            pp[j] += h[j]
            fp, _, _ = qacc_free_of(pp)
            J[:, j] = (fp - f0) / h[j]
        dp, *_ = np.linalg.lstsq(J, -f0, rcond=None)
        scale = 1.0
        p_try = p
        for _ in range(6):
            p_try = p + scale * dp
            f_try, _, _ = qacc_free_of(p_try)
            if np.linalg.norm(f_try) < norm0:
                break
            scale *= 0.5
        p = p_try
        f0 = f_try
        norm0 = np.linalg.norm(f0)
        if norm0 < 0.05:
            break
    f_final, qpos_final, data_final = qacc_free_of(p)
    fz_L = foot_fz(model, data_final, foot_L_id)
    total_weight_N = sum(model.body_mass) * 9.81
    print(f"  최종: p={np.round(p,3)} |qacc_free|={np.linalg.norm(f_final):.5f} "
          f"fz_stance={fz_L:.2f} (목표={total_weight_N:.2f})")
    np.save(out_path, qpos_final)
    print(f"  저장: {out_path}")
    return qpos_final


def main():
    print("=== 좁은(원래) 설정 ===")
    solve_equilibrium(build_narrow_model(), "best_one_leg_qpos_narrow_gn.npy")
    print("=== 현재(넓은) 설정 ===")
    solve_equilibrium(build_current_model(), "best_one_leg_qpos_current_gn.npy")


if __name__ == "__main__":
    main()
