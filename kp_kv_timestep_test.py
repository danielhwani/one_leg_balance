"""kp_kv_timestep_test.py
--------------------------
kp_kv_test.py에서 Kp/Kv를 올릴수록 더 빨리 쓰러지는 걸 봤는데,
Kp가 올라가면 액추에이터의 고유 진동수가 빨라져서 고정 타임스텝
(0.005초)이 그 빠른 동역학을 못 따라가 수치적으로 불안정해지는
것일 수 있다. 타임스텝을 줄여서(0.001초, 5배 세밀하게) 같은
테스트를 다시 돌려본다 - 제어 틱(0.01초)은 그대로 유지하되 서브
스텝 수를 2->10으로 늘림.
"""
import math

import numpy as np
import mujoco

import common
from pelvis_width_test import widen_pelvis
from foot_and_pelvis_test import enlarge_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv


def run_push_fixed_K0_substeps(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, push_mps,
                                axis, n_sub, duration=15.0):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    data.qvel[axis] += push_mps
    mujoco.mj_forward(model, data)
    fell = None
    for tick in range(int(duration / 0.01)):
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


def test_one_config(label, kp_kv_scale, timestep, push_list, axis):
    model = common.load_model()
    widen_pelvis(model, 0.40)
    enlarge_feet(model, 0.24, 0.16)
    if kp_kv_scale != 1.0:
        scale_kp_kv(model, LEG_ACTUATORS, kp_kv_scale)
    model.opt.timestep = timestep
    n_sub = round(0.01 / timestep)
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)

    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    print(f"--- {label} (kp/kv x{kp_kv_scale}, dt={timestep}, n_sub={n_sub}, axis={axis}) ---")
    for push_mps in push_list:
        fell = run_push_fixed_K0_substeps(model, K0, qpos_eq0, qvel_eq0, ctrl_limits,
                                           push_mps, axis, n_sub)
        print(f"  push={push_mps:.2f}: {'생존' if fell is None else f'fell@{fell:.2f}'}")


def main():
    pushes = [0.45, 0.5, 0.6, 0.7, 0.8, 1.0]
    for axis in [0, 1]:
        print(f"========== axis={axis} ({'sagittal' if axis == 0 else 'lateral'}) ==========")
        test_one_config("kp/kv x1.0, dt=0.005(기존)", 1.0, 0.005, pushes, axis=axis)
        test_one_config("kp/kv x2.0, dt=0.005(기존 타임스텝)", 2.0, 0.005, pushes, axis=axis)
        test_one_config("kp/kv x2.0, dt=0.001(5배 세밀)", 2.0, 0.001, pushes, axis=axis)
        test_one_config("kp/kv x3.0, dt=0.001(5배 세밀)", 3.0, 0.001, pushes, axis=axis)
        test_one_config("kp/kv x3.0, dt=0.0005(10배 세밀)", 3.0, 0.0005, pushes, axis=axis)


if __name__ == "__main__":
    main()
