"""step_recovery_v3.py
-----------------------
v2에서 빠졌던 조각: README 교훈 #5("체중을 옮기려면 스탠스(지지)
다리를 조정해야 한다, 들어올릴 다리가 아니다")를 반영한다.

왼발을 들기 전에, 먼저 **오른다리(hip_abduct_L)로 체중을 오른쪽으로
옮기는 램프**(view_weight_shift.py에서 검증된 패턴: 5초 램프 + 5초
정착)를 먼저 하고, 그 다음에야 왼발 스윙(hip_R/knee_R/ankle_pitch_R)
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
        a_abduct_R, a_hip_R, a_knee_R, a_ankle_pitch_R,
        weight_shift_deg, shift_lead_s,
        step_duration_s, lift_knee_deg, land_knee_deg, hip_target_deg,
        relin_interval=0.5, duration_s=20.0, verbose=False):
    """사람처럼: 체중이동과 스윙을 겹쳐서(동시에) 진행한다.
    shift_lead_s만큼 체중이동이 스윙보다 먼저 시작하되(0이면 완전
    동시), 둘 다 step_duration_s 안에서 끝난다 - 미리 멈춰서 기다리는
    단계(투수의 와인드업 같은) 없음."""
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None

    step_t0 = 1.0  # 잠깐 정지 확인 후 바로 시작
    shift_t0 = step_t0 - shift_lead_s
    shift_duration_s = step_duration_s + shift_lead_s
    stepping = False
    landed = False
    ticks_since_land = 0

    K, qpos_eq, qvel_eq, ctrl_trim = K0, qpos_eq0, qvel_eq0, np.zeros(model.nu)

    for tick in range(int(duration_s / 0.01)):
        if not stepping and not landed and data.time >= step_t0:
            stepping = True
            if verbose:
                print(f"  >>> t={data.time:.2f} 왼발 스윙 시작(체중이동과 겹침)")

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

            if not landed and data.time >= shift_t0:
                s_shift = min(1.0, (data.time - shift_t0) / shift_duration_s)
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
                    u[a_hip_R] = hip
                    u[a_knee_R] = knee
                    u[a_ankle_pitch_R] = 0.0

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

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_R")
    a_knee_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_R")
    a_ankle_pitch_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_pitch_R")

    print("=== 체중이동+스윙을 겹쳐서(동시 진행) 시도 ===")
    for weight_shift in [-10, -20]:
        for shift_lead in [0.0, 0.3, 0.6]:
            for step_dur in [0.5, 1.0, 1.5]:
                fell = run(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits,
                           a_abduct_R, a_hip_R, a_knee_R, a_ankle_pitch_R,
                           weight_shift_deg=weight_shift, shift_lead_s=shift_lead,
                           step_duration_s=step_dur, lift_knee_deg=15.0, land_knee_deg=5.0, hip_target_deg=-10.0,
                           duration_s=20.0)
                print(f"shift={weight_shift} lead={shift_lead} step_dur={step_dur}: fell={fell}")


if __name__ == "__main__":
    main()
