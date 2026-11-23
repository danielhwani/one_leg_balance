"""tvlqr_diagnose.py
---------------------
tvlqr_step.py(v5 평형점)가 t=3.58초에 쓰러지는 원인을 더 자세히
본다 - t=3.0초부터 매 틱(0.01초)마다 tilt, 양발 접촉력, 포화
액추에이터 수, 스탠스 다리(로봇 왼쪽) 리언 각도 추적 오차를 찍는다.
"""
import math

import numpy as np
import mujoco

import common
from tvlqr_step import build_waypoints, linearize_at, slerp_lerp_quat


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
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    qpos_eq1 = np.load("best_one_leg_qpos_v5.npy")

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

    T_transition = 4.0
    n_wp = 40
    wps = build_waypoints(qpos_eq0, qpos_eq1, n_wp)
    ctrl_wps = [ctrl_of(q) for q in wps]

    print("웨이포인트별 선형화 중...")
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
    print("완료")

    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    q_abduct_L = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]
    q_waist_roll = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "waist_roll")]
    q_knee_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_R")]
    q_abduct_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_R")]

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None
    for tick in range(int(6.0 / 0.01)):
        t = data.time
        if t < T_transition:
            k = min(n_wp - 1, int(t / dt_wp))
            s_local = (t - k * dt_wp) / dt_wp
            qpos_ref = (1 - s_local) * wps[k] + s_local * wps[k + 1]
            qpos_ref[3:7] = slerp_lerp_quat(wps[k][3:7], wps[k + 1][3:7], s_local)
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
        u_raw = ctrl_ref - K @ x
        u = np.clip(u_raw, -ctrl_limits, ctrl_limits)
        n_sat = int(np.sum(np.abs(u_raw) >= ctrl_limits - 1e-6))
        data.ctrl[:] = u
        for _ in range(2):
            mujoco.mj_step(model, data)

        tilt = common.tilt_deg(data.qpos)
        if t >= 2.9:
            fz_L = foot_fz(model, data, foot_L_id)
            fz_R = foot_fz(model, data, foot_R_id)
            abduct_L_actual = math.degrees(data.qpos[q_abduct_L])
            abduct_L_target = math.degrees(qpos_ref[q_abduct_L])
            waist_roll_actual = math.degrees(data.qpos[q_waist_roll])
            waist_roll_target = math.degrees(qpos_ref[q_waist_roll])
            knee_R_actual = math.degrees(data.qpos[q_knee_R])
            knee_R_target = math.degrees(qpos_ref[q_knee_R])
            abduct_R_actual = math.degrees(data.qpos[q_abduct_R])
            abduct_R_target = math.degrees(qpos_ref[q_abduct_R])
            print(f"t={t:5.2f} tilt={tilt:6.2f} |v|={np.linalg.norm(data.qvel[:3]):.3f} "
                  f"n_sat={n_sat:2d} fz_L={fz_L:7.2f} fz_R={fz_R:7.2f} "
                  f"abdL={abduct_L_actual:6.1f}/{abduct_L_target:6.1f} "
                  f"wroll={waist_roll_actual:6.1f}/{waist_roll_target:6.1f} "
                  f"kneeR={knee_R_actual:6.1f}/{knee_R_target:6.1f} "
                  f"abdR={abduct_R_actual:6.1f}/{abduct_R_target:6.1f}")
        if tilt > 55:
            fell = data.time
            print(f"쓰러짐: t={fell:.2f}")
            break

    if fell is None:
        print("안 쓰러짐(6초)")


if __name__ == "__main__":
    main()
