"""kp_kv_R_test.py
-------------------
지금까지 최선(발+골반 확대 + kp/kv x3.0 + dt=0.0005, 좌우 1.0+ m/s
버팀) 위에서 R도 다시 줄여본다 - 이번엔 제대로 된(세밀한) 타임스텝
기준으로 R 감소의 진짜 효과를 본다. push 범위도 더 올려서(1.0~1.4)
한계를 더 찾아본다.
"""
import math

import numpy as np
import mujoco

import common
from pelvis_width_test import widen_pelvis
from foot_and_pelvis_test import enlarge_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv
from kp_kv_timestep_test import run_push_fixed_K0_substeps


def test_one_config(label, R_scale, push_list, axis):
    model = common.load_model()
    widen_pelvis(model, 0.40)
    enlarge_feet(model, 0.24, 0.16)
    scale_kp_kv(model, LEG_ACTUATORS, 3.0)
    model.opt.timestep = 0.0005
    n_sub = round(0.01 / model.opt.timestep)
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)

    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    R = R * R_scale
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    print(f"--- {label} (R_scale={R_scale}, axis={axis}) ---")
    for push_mps in push_list:
        fell = run_push_fixed_K0_substeps(model, K0, qpos_eq0, qvel_eq0, ctrl_limits,
                                           push_mps, axis, n_sub)
        print(f"  push={push_mps:.2f}: {'생존' if fell is None else f'fell@{fell:.2f}'}")


def main():
    pushes = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8]
    for axis in [0, 1]:
        print(f"========== axis={axis} ({'sagittal' if axis == 0 else 'lateral'}) ==========")
        test_one_config("R 그대로", 1.0, pushes, axis=axis)
        test_one_config("R x2", 2.0, pushes, axis=axis)
        test_one_config("R x5", 5.0, pushes, axis=axis)
        test_one_config("R x10", 10.0, pushes, axis=axis)


if __name__ == "__main__":
    main()
