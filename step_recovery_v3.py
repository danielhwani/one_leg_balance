"""step_recovery_v3.py
-----------------------
v2에서 빠졌던 조각: README 교훈 #5("체중을 옮기려면 스탠스(지지)
다리를 조정해야 한다, 들어올릴 다리가 아니다")를 반영한다.

왼발을 들기 전에, 먼저 **오른다리(hip_abduct_R)로 체중을 오른쪽으로
옮기는 램프**(view_weight_shift.py에서 검증된 패턴: 5초 램프 + 5초
정착)를 먼저 하고, 그 다음에야 왼발 스윙(hip_L/knee_L/ankle_pitch_L)
을 시작한다. push는 당분간 빼고(사용자 지시) 스텝 자체의 안정성만
본다.
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
        a_abduct_R, a_hip_L, a_knee_L, a_ankle_pitch_L,
        weight_shift_deg, ramp_t, pre_settle_t,
        step_duration_s, lift_knee_deg, land_knee_deg, hip_target_deg,
        relin_interval=0.5, duration_s=20.0, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None

    step_t0 = ramp_t + pre_settle_t
    stepping = False
    landed = False
    ticks_since_land = 0

    K, qpos_eq, qvel_eq, ctrl_trim = K0, qpos_eq0, qvel_eq0, np.zeros(model.nu)

    for tick in range(int(duration_s / 0.01)):
        if not stepping and not landed and data.time >= step_t0:
            stepping = True
            if verbose:
                print(f"  >>> t={data.time:.2f} 왼발 스윙 시작")

        if landed and ticks_since_land % int(relin_interval / 0.01) == 0:
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
            u = ctrl_trim - K @ x

            if not landed:
                s_shift = min(1.0, data.time / ramp_t)
                u[a_abduct_R] += math.radians(weight_shift_deg) * s_shift

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
                    u[a_hip_L] = hip
                    u[a_knee_L] = knee
                    u[a_ankle_pitch_L] = 0.0

            u = np.clip(u, -ctrl_limits, ctrl_limits)
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

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    a_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_L")
    a_knee_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_L")
    a_ankle_pitch_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_pitch_L")

    print("=== 체중이동 세기 x 스윙 파라미터 스윕 ===")
    for weight_shift in [-10, -20, -30]:
        for step_dur in [0.5, 1.0, 2.0]:
            for lift_knee in [10, 20]:
                fell = run(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits,
                           a_abduct_R, a_hip_L, a_knee_L, a_ankle_pitch_L,
                           weight_shift_deg=weight_shift, ramp_t=5.0, pre_settle_t=5.0,
                           step_duration_s=step_dur, lift_knee_deg=lift_knee, land_knee_deg=5.0, hip_target_deg=-5.0,
                           duration_s=20.0)
                print(f"shift={weight_shift} step_dur={step_dur} lift_knee={lift_knee}: fell={fell}")


if __name__ == "__main__":
    main()
