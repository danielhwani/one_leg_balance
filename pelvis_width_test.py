"""pelvis_width_test.py
-----------------------
사용자 제안: "골반을 좀 키우면?" - 다리 좌우 간격(leg_L/leg_R의
y offset, 지금 ±0.25m)을 넓히면 균형 범위가 늘어나는지 확인.

예상: 다리가 벌어진 방향은 y(좌우) - 넓히면 **좌우(lateral, 축1)**
push에는 도움이 되지만 **앞뒤(sagittal, 축0)** push엔 발 길이가
그대로라 별 영향이 없을 것. 두 방향 다 테스트해서 가설을 검증한다.

XML 파일은 그대로 두고, 로드한 model 객체의 body_pos만 메모리에서
바꿔서 테스트(파일 수정 없음 - 다른 스크립트에 영향 없음).
"""
import math

import numpy as np
import mujoco

import common
from push_tvlqr import run_push_fixed_K0, run_push_tvlqr


def widen_pelvis(model, half_width):
    leg_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_R")
    leg_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_L")
    model.body_pos[leg_R_id][1] = -half_width
    model.body_pos[leg_L_id][1] = half_width


def test_one_config(label, half_width, push_list, axis):
    model = common.load_model()
    if half_width is not None:
        widen_pelvis(model, half_width)
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
    pushes_sagittal = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    pushes_lateral = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

    print("=== 기존(골반 폭 0.25m) ===")
    test_one_config("기존 폭 0.25", None, pushes_sagittal, axis=0)
    test_one_config("기존 폭 0.25", None, pushes_lateral, axis=1)

    print()
    print("=== 골반 넓힘(0.40m) ===")
    test_one_config("넓힌 폭 0.40", 0.40, pushes_sagittal, axis=0)
    test_one_config("넓힌 폭 0.40", 0.40, pushes_lateral, axis=1)


if __name__ == "__main__":
    main()
