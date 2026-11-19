"""step_recovery_v1.py
-----------------------
외력(push)이 K0의 범위를 넘어서면(예: 0.5m/s), 그 방향으로 한쪽
다리를 빠르게 내딛는 캐치 스텝을 시도한다.

어제 겪은 근본 문제(한 발 서기 자체엔 자연스러운 평형점이 없음 -
무릎만 들어도 2초 안에 쓰러짐, one_leg_balance 프로젝트 초기 README
참고)를 염두에 두고, **최대한 짧게(0.2~0.3초)** 스윙해서 한 발
지지 시간을 최소화하는 전략으로 시작한다.

왼다리(hip_L/knee_L/ankle_pitch_L)만 스크립트로 움직이고, 나머지는
전부 원래 K0(qpos_eq0 기준) - 오른다리가 그동안 전신을 버텨야 한다.
"""
import math

import numpy as np
import mujoco

import common


def run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits,
        a_hip_L, a_knee_L, a_ankle_pitch_L,
        push_mps, push_time_s, step_duration_s, lift_knee_deg, land_knee_deg, hip_target_deg,
        duration_s=15.0, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None
    pushed = False
    stepping = False
    step_t0 = None

    for tick in range(int(duration_s / 0.01)):
        if not pushed and data.time >= push_time_s:
            data.qvel[0] += push_mps
            pushed = True
            stepping = True
            step_t0 = data.time
            if verbose:
                print(f"  >>> t={data.time:.2f} push {push_mps}m/s + 스텝 시작")

        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = np.clip(-K0 @ x, -ctrl_limits, ctrl_limits)

            if stepping:
                s = (data.time - step_t0) / step_duration_s
                if s >= 1.0:
                    stepping = False
                else:
                    if s < 0.5:
                        knee = math.radians(lift_knee_deg) * (s / 0.5)
                    else:
                        knee = math.radians(lift_knee_deg) + (math.radians(land_knee_deg) - math.radians(lift_knee_deg)) * ((s - 0.5) / 0.5)
                    hip = math.radians(hip_target_deg) * min(1.0, s / 0.7)
                    u[a_hip_L] = hip
                    u[a_knee_L] = knee
                    u[a_ankle_pitch_L] = 0.0

            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        t = common.tilt_deg(data.qpos)
        if verbose and tick % 10 == 0:
            print(f"  t={data.time:.2f} tilt={t:.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f} stepping={stepping}")
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
    a_knee_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_L")
    a_ankle_pitch_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_pitch_L")

    print("=== 먼저 스텝 없이, push=0.5m/s가 정말 실패하는지 재확인 ===")
    fell = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_hip_L, a_knee_L, a_ankle_pitch_L,
               0.5, 5.0, 0.01, 0, 0, 0, duration_s=10.0)
    print(f"스텝 없음: fell={fell}")

    print()
    print("=== 캐치 스텝 파라미터 스윕 (push=0.5m/s) ===")
    for step_dur in [0.15, 0.2, 0.3]:
        for lift_knee in [20, 30, 40]:
            for hip_target in [-15, -25, -35]:
                fell = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_hip_L, a_knee_L, a_ankle_pitch_L,
                           0.5, 5.0, step_dur, lift_knee, 10.0, hip_target, duration_s=10.0)
                print(f"step_dur={step_dur} lift_knee={lift_knee} hip_target={hip_target}: fell={fell}")


if __name__ == "__main__":
    main()
