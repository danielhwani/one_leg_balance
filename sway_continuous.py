"""sway_continuous.py
----------------------
"같은 자리에서 계속 좌우로 살살 흔들리는" 진짜 물리적 흔들림 -
불연속 전환(기울고-멈추고-돌아오고) 대신, hip_abduct 목표를 사인파
처럼 부드럽게 좌우로 계속 오가게 한다. 진폭을 K0의 실제 유효 범위
(±3~4도) 안에서만 쓰면, 범위를 벗어난 적이 없으니 재선형화도 복귀
단계도 필요 없이 **처음부터 끝까지 원래 K0 하나로** 계속 흔들 수
있는지 확인한다.
"""
import math

import numpy as np
import mujoco

import common


def run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_abduct_R, a_abduct_L,
        amplitude_deg, period_s, duration_s, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None

    for tick in range(int(duration_s / 0.01)):
        bias = amplitude_deg * math.sin(2 * math.pi * data.time / period_s)
        bias_rad = math.radians(bias)
        hip_L_cmd = min(bias_rad, 0.0)  # 음수 -> 오른쪽
        hip_R_cmd = max(bias_rad, 0.0)  # 양수 -> 왼쪽
        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = -K0 @ x
            u[a_abduct_R] = hip_L_cmd
            u[a_abduct_L] = hip_R_cmd
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


def main():
    model = common.load_model()
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")

    print("=== amp=3deg period=1.0s(걸음걸이 수준), 150초 장시간 검증 ===")
    fell = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_abduct_R, a_abduct_L,
               3.0, 1.0, 150.0, verbose=True)
    print(f"fell={fell}")


if __name__ == "__main__":
    main()
