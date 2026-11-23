"""kp_kv_test.py
-----------------
사용자 제안: "액추에이터 Kp, Kv를 강화하는 것은 어때?" - R(제어
쪽에서 얼마나 적극적으로 쓸지)과 달리 Kp/Kv는 액추에이터 자체의
실제 힘(하드웨어 스펙)이다. 다리 액추에이터(hip/knee/ankle)의
kp/kv를 올려서 발+골반 확대(0.45 m/s, 지금까지 최선) 위에 추가로
테스트.

position 액추에이터의 kp는 gainprm[0]과 biasprm[1](=-kp)에, kv는
biasprm[2](=-kv)에 들어있다 - 둘 다 같이 스케일.
"""
import math

import numpy as np
import mujoco

import common
from push_tvlqr import run_push_fixed_K0, run_push_tvlqr
from pelvis_width_test import widen_pelvis
from foot_and_pelvis_test import enlarge_feet


LEG_ACTUATORS = [
    "act_hip_abduct_R", "act_hip_abduct_L", "act_hip_R", "act_hip_L",
    "act_knee_R", "act_knee_L",
    "act_ankle_roll_R", "act_ankle_roll_L", "act_ankle_pitch_R", "act_ankle_pitch_L",
]


def scale_kp_kv(model, actuator_names, scale):
    for name in actuator_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        model.actuator_gainprm[aid, 0] *= scale
        model.actuator_biasprm[aid, 1] *= scale
        model.actuator_biasprm[aid, 2] *= scale


def test_one_config(label, kp_kv_scale, push_list, axis):
    model = common.load_model()
    widen_pelvis(model, 0.40)
    enlarge_feet(model, 0.24, 0.16)
    if kp_kv_scale != 1.0:
        scale_kp_kv(model, LEG_ACTUATORS, kp_kv_scale)
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)

    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    print(f"--- {label} (kp/kv x{kp_kv_scale}, axis={axis}) ---")
    for push_mps in push_list:
        fell_fixed = run_push_fixed_K0(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, push_mps, axis=axis)
        fell_tv = run_push_tvlqr(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits, push_mps, axis=axis)
        print(f"  push={push_mps:.2f}: 고정K0={'생존' if fell_fixed is None else f'fell@{fell_fixed:.2f}'}  "
              f"TV-LQR={'생존' if fell_tv is None else f'fell@{fell_tv:.2f}'}")


def main():
    pushes = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
    for axis in [0, 1]:
        print(f"========== axis={axis} ({'sagittal' if axis == 0 else 'lateral'}) ==========")
        test_one_config("kp/kv 그대로", 1.0, pushes, axis=axis)
        test_one_config("kp/kv x1.5", 1.5, pushes, axis=axis)
        test_one_config("kp/kv x2.0", 2.0, pushes, axis=axis)
        test_one_config("kp/kv x3.0", 3.0, pushes, axis=axis)


if __name__ == "__main__":
    main()
