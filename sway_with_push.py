"""sway_with_push.py
---------------------
좌우로 계속 흔드는(sway_continuous.py) 도중에 외력(push, 속도
교란)을 추가로 주면 버티는지 확인. 혼자일 때는 push 0.25m/s까지
버텼는데(compare_offset_vs_push.py), 흔드는 중이면 이미 상태가
약간씩 벗어나 있어서 여유가 줄어들 수 있다.
"""
import math

import numpy as np
import mujoco

import common


def run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_abduct_R, a_abduct_L,
        amplitude_deg, period_s, push_mps, push_time_s, duration_s, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None
    pushed = False

    for tick in range(int(duration_s / 0.01)):
        if not pushed and data.time >= push_time_s:
            data.qvel[0] += push_mps
            pushed = True
            if verbose:
                print(f"  >>> t={data.time:.2f} push {push_mps}m/s 주입")

        bias = amplitude_deg * math.sin(2 * math.pi * data.time / period_s)
        bias_rad = math.radians(bias)
        hip_R_cmd = min(bias_rad, 0.0)
        hip_L_cmd = max(bias_rad, 0.0)
        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = -K0 @ x
            u[a_abduct_R] = hip_R_cmd
            u[a_abduct_L] = hip_L_cmd
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        t = common.tilt_deg(data.qpos)
        if verbose and tick % 20 == 0:
            print(f"  t={data.time:.2f} bias={bias:.2f}deg tilt={t:.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f}")
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

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")

    AMP, PERIOD = 3.0, 1.0

    print("=== push=0.3m/s, 1초 사이클을 10개 구간으로 세밀하게 ===")
    for i in range(10):
        push_time = 5.0 + i * 0.1
        fell = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_abduct_R, a_abduct_L,
                   AMP, PERIOD, 0.3, push_time, 20.0)
        print(f"push_time={push_time:.1f}s (위상={i*36}도): fell={fell}")


if __name__ == "__main__":
    main()
