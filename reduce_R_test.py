"""reduce_R_test.py
--------------------
머리/가슴 무게 변경은 되돌리고(발+골반 확대만 유지, 지금까지 최선
0.45 m/s), R(액추에이터 힘 페널티)을 줄여서 더 적극적인 제어를
허용하면 push 복원력이 더 늘어나는지 확인.
"""
import math

import numpy as np
import mujoco

import common
from push_tvlqr import run_push_fixed_K0, run_push_tvlqr
from pelvis_width_test import widen_pelvis
from foot_and_pelvis_test import enlarge_feet


def test_one_config(label, R_scale, push_list, axis):
    model = common.load_model()
    widen_pelvis(model, 0.40)
    enlarge_feet(model, 0.24, 0.16)
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)

    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    R = R * R_scale
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    print(f"--- {label} (R_scale={R_scale}, axis={axis}) ---")
    for push_mps in push_list:
        fell_fixed = run_push_fixed_K0(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, push_mps, axis=axis)
        fell_tv = run_push_tvlqr(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits, push_mps, axis=axis)
        print(f"  push={push_mps:.2f}: 고정K0={'생존' if fell_fixed is None else f'fell@{fell_fixed:.2f}'}  "
              f"TV-LQR={'생존' if fell_tv is None else f'fell@{fell_tv:.2f}'}")


def main():
    pushes = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
    for axis in [0, 1]:
        print(f"========== axis={axis} ({'sagittal' if axis == 0 else 'lateral'}) ==========")
        test_one_config("R 그대로(기존)", 1.0, pushes, axis=axis)
        test_one_config("R x0.5", 0.5, pushes, axis=axis)
        test_one_config("R x0.2", 0.2, pushes, axis=axis)
        test_one_config("R x0.1", 0.1, pushes, axis=axis)


if __name__ == "__main__":
    main()
