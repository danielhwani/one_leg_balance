"""sway_forward_back.py
------------------------
좌우 흔들기(sway_continuous.py)와 같은 방식, 축만 hip_abduct(좌우)
대신 hip_L/hip_R(피치, 양쪽 다리를 같은 방향으로 - 앞뒤로 숙이는
자연스러운 동작)을 쓴다. check_pitch_basin.py로 범위(+6도/-5도)를
먼저 확인했으므로 진폭은 4도로 안전하게.
"""
import math

import numpy as np
import mujoco

import common


def run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_hip_L, a_hip_R,
        amplitude_deg, period_s, duration_s, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None

    for tick in range(int(duration_s / 0.01)):
        bias_rad = math.radians(amplitude_deg * math.sin(2 * math.pi * data.time / period_s))
        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = -K0 @ x
            u[a_hip_L] = bias_rad
            u[a_hip_R] = bias_rad
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
    model = common.load_model()
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_L")
    a_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_R")

    print("=== amp=4deg period=1.0s(걸음걸이 수준), 150초 장시간 검증 ===")
    fell = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_hip_L, a_hip_R,
               4.0, 1.0, 150.0, verbose=True)
    print(f"fell={fell}")


if __name__ == "__main__":
    main()
