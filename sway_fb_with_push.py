import math

import numpy as np
import mujoco

import common


def run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_hip_R, a_hip_L,
        amplitude_deg, period_s, push_axis, push_mps, push_time_s, duration_s, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None
    pushed = False

    for tick in range(int(duration_s / 0.01)):
        if not pushed and data.time >= push_time_s:
            data.qvel[push_axis] += push_mps
            pushed = True

        bias_rad = math.radians(amplitude_deg * math.sin(2 * math.pi * data.time / period_s))
        for _ in range(2):
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
        if verbose and tick % 20 == 0:
            print(f"  t={data.time:.2f} tilt={t:.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f}")
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

    a_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_R")
    a_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_L")

    AMP, PERIOD = 4.0, 1.0

    print("=== 앞뒤 흔드는 중 전진(X) push, 사이클 10개 구간 ===")
    for i in range(10):
        push_time = 5.0 + i * 0.1
        fell = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_hip_R, a_hip_L,
                   AMP, PERIOD, 0, 0.3, push_time, 20.0)
        print(f"push_time={push_time:.1f}s (위상={i*36}도): fell={fell}")

    print()
    print("=== 앞뒤 흔드는 중 전진(X) push 세기 스윕 (위상=0도) ===")
    for push in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        fell = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_hip_R, a_hip_L,
                   AMP, PERIOD, 0, push, 5.0, 20.0)
        print(f"push={push}m/s: fell={fell}")

    print()
    print("=== 앞뒤 흔드는 중 좌우(Y) push 세기 스윕 (위상=0도) ===")
    for push in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        fell = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_hip_R, a_hip_L,
                   AMP, PERIOD, 1, push, 5.0, 20.0)
        print(f"push={push}m/s: fell={fell}")


if __name__ == "__main__":
    main()
