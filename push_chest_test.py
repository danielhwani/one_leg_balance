"""push_chest_test.py
----------------------
지금까지의 push 테스트는 data.qvel[0:3] += push_mps로 자유관절(몸
전체) 속도를 한 번에 바꿔서 - 실제로 어떤 부위를 미는 게 아니라
회전(토크) 효과가 전혀 없는 "순수 평행이동" 외력이었다. 실제로
가슴을 밀면 무게중심에서 떨어진 위치라 회전까지 같이 생긴다.

이번엔 xfrc_applied로 가슴(chest) 바디에 짧은 시간(0.1초) 동안
실제 힘(N)을 가해서 더 현실적으로 테스트한다 - 힘 x 시간(충격량)이
기존의 "속도 변화 x 전체질량"과 같아지도록 보정해서 직접 비교
가능하게 한다.
"""
import math

import numpy as np
import mujoco

import common
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


def run_chest_push(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, chest_id,
                    force_vec, push_duration, n_sub, duration=15.0):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None
    for tick in range(int(duration / 0.01)):
        if data.time < push_duration:
            data.xfrc_applied[chest_id, 0:3] = force_vec
        else:
            data.xfrc_applied[chest_id, 0:3] = 0.0
        for _ in range(n_sub):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = np.clip(-K0 @ x, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        t = common.tilt_deg(data.qpos)
        if t > 55 and fell is None:
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

    chest_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chest")
    total_mass = sum(model.body_mass)
    push_duration = 0.1

    print(f"가슴 바디 id={chest_id}, 전체 질량={total_mass:.2f}kg, "
          f"충격 지속시간={push_duration}s")
    print("(기존 push_mps와 비교: impulse = push_mps * total_mass, force = impulse/duration)")

    # 등가 push_mps 기준으로 힘 스윕 (impulse = push_mps * total_mass)
    equiv_push_mps_list = [0.3, 0.5, 0.7, 0.75, 0.8, 1.0, 1.05, 1.2]
    for axis, axis_name in [(0, "sagittal(가슴 앞으로 밀기, +X)"), (1, "lateral(가슴 옆으로 밀기, +Y)")]:
        print(f"=== {axis_name} ===")
        for push_mps in equiv_push_mps_list:
            force_mag = push_mps * total_mass / push_duration
            force_vec = np.zeros(3)
            force_vec[axis] = force_mag
            fell = run_chest_push(model, K0, qpos_eq0, qvel_eq0, ctrl_limits,
                                   chest_id, force_vec, push_duration, n_sub)
            print(f"  등가 push={push_mps:.2f} m/s (힘={force_mag:.1f}N x {push_duration}s): "
                  f"{'생존' if fell is None else f'fell@{fell:.2f}'}")


if __name__ == "__main__":
    main()
