"""reduce_upper_mass_test.py
-----------------------------
사용자 제안: "머리와 가슴의 무게를 줄일 수도 있지?" - 상체(가슴
9kg, 머리 4kg) 질량을 줄여서 무게중심을 낮추고 관성을 줄이면
push 복원력이 더 늘어나는지 확인. 발+골반 확대(지금까지 최선,
0.45 m/s)에 추가로 테스트.

body_mass를 직접 줄이면서 body_inertia도 같은 비율로 스케일 -
모양은 그대로 두고 가볍게만 만드는 것(물리적으로 일관되게).
"""
import math

import numpy as np
import mujoco

import common
from push_tvlqr import run_push_fixed_K0, run_push_tvlqr
from pelvis_width_test import widen_pelvis
from foot_and_pelvis_test import enlarge_feet


def scale_body_mass(model, body_name, new_mass):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    old_mass = model.body_mass[bid]
    factor = new_mass / old_mass
    model.body_mass[bid] = new_mass
    model.body_inertia[bid] *= factor
    return old_mass, new_mass


def test_one_config(label, chest_mass, head_mass, push_list, axis):
    model = common.load_model()
    widen_pelvis(model, 0.40)
    enlarge_feet(model, 0.24, 0.16)
    if chest_mass is not None:
        scale_body_mass(model, "chest", chest_mass)
    if head_mass is not None:
        scale_body_mass(model, "head", head_mass)
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    pelvis_z = qpos_eq0[2]
    total_mass = sum(model.body_mass)

    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    print(f"--- {label}: pelvis_z={pelvis_z:.3f} total_mass={total_mass:.2f} (axis={axis}) ---")
    for push_mps in push_list:
        fell_fixed = run_push_fixed_K0(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, push_mps, axis=axis)
        fell_tv = run_push_tvlqr(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits, push_mps, axis=axis)
        print(f"  push={push_mps:.2f}: 고정K0={'생존' if fell_fixed is None else f'fell@{fell_fixed:.2f}'}  "
              f"TV-LQR={'생존' if fell_tv is None else f'fell@{fell_tv:.2f}'}")


def main():
    pushes = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    for axis in [0, 1]:
        print(f"========== axis={axis} ({'sagittal' if axis == 0 else 'lateral'}) ==========")
        test_one_config("발+골반만(기존 질량)", None, None, pushes, axis=axis)
        test_one_config("가슴 9->5, 머리 4->2", 5.0, 2.0, pushes, axis=axis)


if __name__ == "__main__":
    main()
