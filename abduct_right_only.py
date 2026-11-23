"""abduct_right_only.py
------------------------
가설 검증: 다리를 벌리면(abduct) 그 다리의 지지력이 줄어들면서
반대쪽 다리로 체중이 자동으로 쏠리는가? 오른쪽 다리(hip_abduct_R)
만 벌려보고(왼쪽 다리는 아무 명령도 주지 않음) 왼발(foot_L)에
체중이 쏠리는지, 그리고 정강이-땅 접촉 버그를 고친 모델에서
60초간 안정적인지 확인한다.
"""
import math

import numpy as np
import mujoco

import common
from pelvis_width_test import widen_pelvis
from foot_offset_test import set_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv


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


def run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub, a_abduct_R, q_abduct_R,
        foot_L_id, foot_R_id, target_deg, ramp_t=5.0, duration_s=60.0, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None

    for tick in range(int(duration_s / 0.01)):
        s = min(1.0, data.time / ramp_t)
        ff_R = math.radians(target_deg) * s
        for _ in range(n_sub):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = -K0 @ x
            u[a_abduct_R] += ff_R
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        tilt = common.tilt_deg(data.qpos)
        if verbose and tick % 200 == 0:
            fz_L = foot_fz(model, data, foot_L_id)
            fz_R = foot_fz(model, data, foot_R_id)
            achieved_deg = math.degrees(data.qpos[q_abduct_R] - qpos_eq0[q_abduct_R])
            print(f"  t={data.time:6.2f} tilt={tilt:5.2f} fz_L={fz_L:6.1f} fz_R={fz_R:6.1f} "
                  f"achieved={achieved_deg:5.2f}deg(목표={target_deg*s:.2f})")
        if tilt > 55 and fell is None:
            fell = data.time
            break
    return fell


def main():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.28, 0.16, 0.0)
    scale_kp_kv(model, LEG_ACTUATORS, 3.0)
    model.opt.timestep = 0.0005
    n_sub = round(0.01 / model.opt.timestep)
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    R = R * 2.0
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    q_abduct_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_R")]
    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")

    print("=== 오른쪽 다리만 벌림(hip_abduct_R) - 왼발에 체중이 쏠리는지 확인, 60초씩 ===")
    for target_deg in [4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0]:
        fell = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub, a_abduct_R, q_abduct_R,
                   foot_L_id, foot_R_id, target_deg, duration_s=60.0,
                   verbose=(target_deg == 12.0))
        print(f"목표={target_deg:5.1f}도: {'생존(60초)' if fell is None else f'fell@{fell:.2f}'}")


if __name__ == "__main__":
    main()
