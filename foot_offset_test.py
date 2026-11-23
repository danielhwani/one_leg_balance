"""foot_offset_test.py
-----------------------
발 box geom이 발목(ankle) 기준 앞으로 0.06 오프셋되어 있어서
(앞 0.30m, 뒤 0.18m 비대칭) 뒤로 넘어지는 쪽이 약하다. 오프셋을
0(발목 중심 대칭)으로 바꿔서 앞/뒤 push 한계가 균형 잡히는지
확인한다. 지금까지 최선 구성(발+골반+kp/kv x3+dt=0.0005+R x2) 위에.
"""
import math

import numpy as np
import mujoco

import common
from pelvis_width_test import widen_pelvis
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv
from kp_kv_timestep_test import run_push_fixed_K0_substeps


def set_feet(model, size_x, size_y, pos_x):
    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    for fid in [foot_L_id, foot_R_id]:
        g = model.body_geomadr[fid]
        model.geom_size[g][0] = size_x
        model.geom_size[g][1] = size_y
        model.geom_pos[g][0] = pos_x


def test_one_config(label, pos_x, push_list, axis):
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.24, 0.16, pos_x)
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

    print(f"--- {label} (foot pos_x={pos_x}, axis={axis}) ---")
    for push_mps in push_list:
        fell = run_push_fixed_K0_substeps(model, K0, qpos_eq0, qvel_eq0, ctrl_limits,
                                           push_mps, axis, n_sub)
        print(f"  push={push_mps:+.2f}: {'생존' if fell is None else f'fell@{fell:.2f}'}")


def test_size(label, size_x, pushes, axis=0):
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, size_x, 0.16, 0.0)
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
    print(f"--- {label}: 발 길이(half)={size_x} (전체 {2*size_x:.2f}m), axis={axis} ---")
    for push_mps in pushes:
        fell = run_push_fixed_K0_substeps(model, K0, qpos_eq0, qvel_eq0, ctrl_limits,
                                           push_mps, axis, n_sub)
        print(f"  push={push_mps:+.2f}: {'생존' if fell is None else f'fell@{fell:.2f}'}")


def main():
    pushes_fwd = [0.7, 0.8, 0.9, 1.0, 1.1]
    pushes_bwd = [-0.7, -0.8, -0.9, -1.0, -1.1]
    for size_x in [0.26, 0.28, 0.30, 0.32]:
        test_size(f"발 길이(half)={size_x}, 대칭", size_x, pushes_fwd, axis=0)
        test_size(f"발 길이(half)={size_x}, 대칭", size_x, pushes_bwd, axis=0)


if __name__ == "__main__":
    main()
