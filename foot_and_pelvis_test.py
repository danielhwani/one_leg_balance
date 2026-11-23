"""foot_and_pelvis_test.py
---------------------------
골반 폭 넓히기(0.25->0.40m)가 push 복원 범위를 늘린다는 걸 확인했다
(좌우 0.25->0.40, 앞뒤 0.25->0.30 m/s). 이번엔 발 크기도 키워서
같이/따로 테스트한다 - 발 자체가 커지면 ZMP 여유가 더 늘어날 것으로
예상(특히 길이를 늘리면 앞뒤, 폭을 늘리면 좌우에 도움).

발 box geom 크기(half-extent): 기존 0.17 x 0.10 x 0.025 -> 0.24 x
0.16 x 0.025 (길이 +41%, 폭 +60%)로 키운다. XML은 그대로 두고
로드한 model의 geom_size/geom_pos만 메모리에서 바꿔서 테스트.
"""
import math

import numpy as np
import mujoco

import common
from push_tvlqr import run_push_fixed_K0, run_push_tvlqr
from pelvis_width_test import widen_pelvis


def enlarge_feet(model, size_x, size_y):
    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    for fid in [foot_L_id, foot_R_id]:
        g = model.body_geomadr[fid]
        model.geom_size[g][0] = size_x
        model.geom_size[g][1] = size_y


def test_one_config(label, widen_to, foot_xy, push_list, axis):
    model = common.load_model()
    if widen_to is not None:
        widen_pelvis(model, widen_to)
    if foot_xy is not None:
        enlarge_feet(model, *foot_xy)
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    print(f"--- {label} (axis={axis}) ---")
    for push_mps in push_list:
        fell_fixed = run_push_fixed_K0(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, push_mps, axis=axis)
        fell_tv = run_push_tvlqr(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits, push_mps, axis=axis)
        print(f"  push={push_mps:.2f}: 고정K0={'생존' if fell_fixed is None else f'fell@{fell_fixed:.2f}'}  "
              f"TV-LQR={'생존' if fell_tv is None else f'fell@{fell_tv:.2f}'}")


def main():
    pushes = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    FOOT_XY = (0.24, 0.16)

    for axis in [0, 1]:
        print(f"========== axis={axis} ({'sagittal(앞뒤)' if axis == 0 else 'lateral(좌우)'}) ==========")
        test_one_config("발만 키움(0.24x0.16)", None, FOOT_XY, pushes, axis=axis)
        test_one_config("골반+발 둘 다", 0.40, FOOT_XY, pushes, axis=axis)


if __name__ == "__main__":
    main()
