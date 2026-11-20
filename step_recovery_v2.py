"""step_recovery_v2.py
-----------------------
v1 수정판 - 사용자 통찰: "사람은 한 발로 균형을 잡은 후 스텝을
밟는 게 아니라, 넘어지지 않으려고 빠르게 스텝을 내딛는다." 즉
스텝 "도중"에 불안정한 건 정상이고, 중요한 건 **착지 후 그 운동량을
제대로 받아내는지**다.

v1은 착지 후에도 계속 원래(양발 대칭) qpos_eq0 기준 K0를 썼는데,
착지 자세는 한쪽 다리가 앞으로 나가 있어서 qpos_eq0와 많이
다르다 - 오늘 배운 "K0는 범위를 벗어나면 못 버틴다"가 착지 후에도
또 발생했을 수 있다. 그래서 착지 직후 **그 자세에서 다시
재선형화**하고(오늘 흔들기에서 쓴 것과 같은 패턴), 1초마다 계속
갱신하며 유지해본다.
"""
import math

import numpy as np
import mujoco

import common


def relinearize_at(model, qpos, ctrl, Q, R):
    nv, nu = model.nv, model.nu
    data_lin = mujoco.MjData(model)
    data_lin.qpos[:] = qpos
    data_lin.qvel[:] = 0.0
    data_lin.ctrl[:] = ctrl
    mujoco.mj_forward(model, data_lin)
    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data_lin, 1e-6, True, A, B, None, None)
    return common.solve_dare(A, B, Q, R)


def run(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits,
        a_hip_R, a_knee_R, a_ankle_pitch_R,
        push_mps, push_time_s, step_duration_s, lift_knee_deg, land_knee_deg, hip_target_deg,
        duration_s=15.0, relin_interval=1.0, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None
    pushed = False
    stepping = False
    landed = False
    step_t0 = None

    K, qpos_eq, qvel_eq, ctrl_trim = K0, qpos_eq0, qvel_eq0, np.zeros(model.nu)
    RELIN_INTERVAL = relin_interval
    ticks_since_land = 0

    for tick in range(int(duration_s / 0.01)):
        if not pushed and data.time >= push_time_s:
            data.qvel[0] += push_mps
            pushed = True
            stepping = True
            step_t0 = data.time
            if verbose:
                print(f"  >>> t={data.time:.2f} push {push_mps}m/s + 스텝 시작")

        if landed and ticks_since_land % int(RELIN_INTERVAL / 0.01) == 0:
            qpos_eq = data.qpos.copy()
            qvel_eq = np.zeros_like(qvel_eq0)
            ctrl_trim = data.ctrl.copy()
            K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
        if landed:
            ticks_since_land += 1

        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
            dv = data.qvel - qvel_eq
            x = np.concatenate([dq, dv])
            u = np.clip(ctrl_trim - K @ x, -ctrl_limits, ctrl_limits)

            if stepping:
                s = (data.time - step_t0) / step_duration_s
                if s >= 1.0:
                    stepping = False
                    landed = True
                    ticks_since_land = 0
                else:
                    if s < 0.5:
                        knee = math.radians(lift_knee_deg) * (s / 0.5)
                    else:
                        knee = math.radians(lift_knee_deg) + (math.radians(land_knee_deg) - math.radians(lift_knee_deg)) * ((s - 0.5) / 0.5)
                    hip = math.radians(hip_target_deg) * min(1.0, s / 0.7)
                    u[a_hip_R] = hip
                    u[a_knee_R] = knee
                    u[a_ankle_pitch_R] = 0.0

            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        t = common.tilt_deg(data.qpos)
        if verbose and tick % 10 == 0:
            print(f"  t={data.time:.2f} tilt={t:.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f} "
                  f"stepping={stepping} landed={landed}")
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
    a_knee_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_R")
    a_ankle_pitch_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_pitch_R")

    print("=== 가장 약한 조합 상세 추적 (step_dur=1.5, lift_knee=10, hip_target=-5) ===")
    fell = run(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits, a_hip_R, a_knee_R, a_ankle_pitch_R,
               0.0, 5.0, 1.5, 10.0, 5.0, -5.0, duration_s=15.0, relin_interval=0.5, verbose=True)
    print(f"fell={fell}")


if __name__ == "__main__":
    main()
