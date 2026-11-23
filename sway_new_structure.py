"""sway_new_structure.py
-------------------------
외력(push)이 아니라 스스로 좌우/앞뒤로 계속 흔드는 것(sway_continuous.py
/ sway_forward_back.py와 같은 방식)을 오늘 최종 구성(발+골반 확대
+ Kp/Kv x3 + 타임스텝 0.0005 + R x2) 위에서 다시 확인한다 - 구조와
게인이 바뀌었으니 예전에 검증된 진폭(좌우 3도, 앞뒤 4도)이 여전히
안전한지 먼저 헤드리스로 본다.
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


def run_lateral(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_abduct_L, a_abduct_R,
                 amplitude_deg, period_s, duration_s, n_sub, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None
    for tick in range(int(duration_s / 0.01)):
        bias = amplitude_deg * math.sin(2 * math.pi * data.time / period_s)
        bias_rad = math.radians(bias)
        hip_L_cmd = min(bias_rad, 0.0)
        hip_R_cmd = max(bias_rad, 0.0)
        for _ in range(n_sub):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = -K0 @ x
            u[a_abduct_L] = hip_L_cmd
            u[a_abduct_R] = hip_R_cmd
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        t = common.tilt_deg(data.qpos)
        if verbose and tick % 100 == 0:
            print(f"  t={data.time:.2f} bias={bias:.2f}deg tilt={t:.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f}")
        if t > 55 and fell is None:
            fell = data.time
            break
    return fell


def run_fb(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_hip_R, a_hip_L,
           amplitude_deg, period_s, duration_s, n_sub, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None
    for tick in range(int(duration_s / 0.01)):
        bias_rad = math.radians(amplitude_deg * math.sin(2 * math.pi * data.time / period_s))
        for _ in range(n_sub):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = -K0 @ x
            u[a_hip_R] = bias_rad
            u[a_hip_L] = bias_rad
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        t = common.tilt_deg(data.qpos)
        if verbose and tick % 100 == 0:
            print(f"  t={data.time:.2f} bias={math.degrees(bias_rad):.2f}deg tilt={t:.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f}")
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

    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    a_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_L")
    a_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_R")

    print("=== 좌우 흔들기 amp=3deg period=1.0s, 30초 ===")
    fell = run_lateral(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_abduct_L, a_abduct_R,
                        3.0, 1.0, 30.0, n_sub, verbose=True)
    print(f"fell={fell}")

    print()
    print("=== 앞뒤 흔들기 amp=4deg period=1.0s, 30초 ===")
    fell = run_fb(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_hip_R, a_hip_L,
                  4.0, 1.0, 30.0, n_sub, verbose=True)
    print(f"fell={fell}")


if __name__ == "__main__":
    main()
