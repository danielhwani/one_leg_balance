"""abduct_red_negative.py
--------------------------
빨간 다리(foot_R)를 "진짜로" 바깥쪽으로 벌리는 방향(음수 각도,
방금 좌표로 확인됨)으로, 더 작은 각도부터 60초씩 길게 테스트한다.

용어(혼동 방지): 코드상 foot_R = 빨간 다리, foot_L = 파란 다리.
양수 hip_abduct_R은 안쪽으로 좁혀지는 잘못된 방향이었고, 음수가
실제로 바깥쪽으로 벌어지는 방향(확인됨: -8도에서 foot_R_y가
-0.40->-0.50로 이동, 바깥쪽).
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


def run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub, a_abduct_R,
        foot_red_id, foot_blue_id, target_deg, ramp_t=5.0, duration_s=60.0, verbose=False):
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
        if verbose and tick % 300 == 0:
            fz_red = foot_fz(model, data, foot_red_id)
            fz_blue = foot_fz(model, data, foot_blue_id)
            red_y = data.xpos[foot_red_id][1]
            print(f"  t={data.time:6.2f} tilt={tilt:5.2f} fz_빨강={fz_red:6.1f} fz_파랑={fz_blue:6.1f} "
                  f"빨강발_y={red_y:.4f}")
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
    foot_red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")   # 빨간 다리
    foot_blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")  # 파란 다리

    print("=== 빨간 다리(foot_R) 바깥으로 벌리기(음수 각도) - 60초씩, 작은 각도부터 ===")
    for target_deg in [-2.0, -4.0, -6.0, -8.0, -10.0, -12.0]:
        fell = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub, a_abduct_R,
                   foot_red_id, foot_blue_id, target_deg, duration_s=60.0,
                   verbose=(target_deg == -6.0))
        print(f"목표={target_deg:6.1f}도: {'생존(60초)' if fell is None else f'fell@{fell:.2f}'}")


if __name__ == "__main__":
    main()
