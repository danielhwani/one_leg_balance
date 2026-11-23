"""weight_shift_left_only.py
-----------------------------
스텝을 다시 시작 - 이번엔 체중이동(왼발, foot_L, 파란 다리)만
따로 떼서 충분히 길게(60초+) 테스트한다. 다리 벌리기는 이게
확실히 검증된 다음에 추가한다.

오늘 교훈: 짧게(10~15초) 보고 안정적이라고 판단하면 안 된다 -
반드시 길게(60초) 돌려본다. hip_abduct_L에 +각도(왼발로 체중)를
여러 크기로 램프+유지해서, 얼마나 큰 각도까지 "진짜로" 오래
버티는지 확인한다.
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


def build_best_model():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.28, 0.16, 0.0)
    scale_kp_kv(model, LEG_ACTUATORS, 3.0)
    model.opt.timestep = 0.0005
    return model


def run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub, a_abduct_L, q_abduct_L,
        foot_L_id, foot_R_id, target_deg, ramp_t=5.0, duration_s=60.0, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None

    for tick in range(int(duration_s / 0.01)):
        s = min(1.0, data.time / ramp_t)
        ff_L = math.radians(target_deg) * s
        for _ in range(n_sub):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = -K0 @ x
            u[a_abduct_L] += ff_L
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        tilt = common.tilt_deg(data.qpos)
        if verbose and tick % 200 == 0:
            fz_L = foot_fz(model, data, foot_L_id)
            fz_R = foot_fz(model, data, foot_R_id)
            achieved_deg = math.degrees(data.qpos[q_abduct_L] - qpos_eq0[q_abduct_L])
            print(f"  t={data.time:6.2f} tilt={tilt:5.2f} fz_L={fz_L:6.1f} fz_R={fz_R:6.1f} "
                  f"achieved={achieved_deg:5.2f}deg(목표={target_deg*s:.2f})")
        if tilt > 55 and fell is None:
            fell = data.time
            break
    return fell


def main():
    model = build_best_model()
    n_sub = round(0.01 / model.opt.timestep)
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    R = R * 2.0
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    q_abduct_L = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]
    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")

    print("=== 왼발(foot_L, 파란 다리)로 체중이동만 - 60초씩 여러 각도 ===")
    for target_deg in [4.0, 8.0, 12.0, 16.0, 20.0]:
        fell = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub, a_abduct_L, q_abduct_L,
                   foot_L_id, foot_R_id, target_deg, duration_s=60.0,
                   verbose=(target_deg == 8.0))
        print(f"목표={target_deg:5.1f}도: {'생존(60초)' if fell is None else f'fell@{fell:.2f}'}")


if __name__ == "__main__":
    main()
