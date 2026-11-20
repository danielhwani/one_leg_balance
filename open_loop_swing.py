"""open_loop_swing.py
---------------------
사용자 질문: "발을 들어서 착지할 때까지 제어가 필요한 상황이니?
그냥 이동으로 느껴지는데?"

지금까지의 6번 시도는 전부 스윙 중에도 전신에 K0(또는 재선형화된
K)를 계속 걸어놨다 - 스윙 다리 외 모든 관절이 "원래 서있던 자세로
돌아가라"는 피드백을 계속 받는다. 체중이 지지발 쪽으로 쏠리는 건
자연스러운 결과인데, K0 입장에선 그 쏠림 자체가 에러라서 되돌리려고
적극적으로 싸운다 - 그 싸움이 이륙 순간의 급격한 속도 점프의 원인일
수 있다.

이 스크립트는 다르게 해본다: 스윙 구간(이륙~착지) 동안은 전신
피드백(K@x)을 끄고, 모든 액추에이터를 "이륙 시점의 trim 값에
고정"(로컬 position-PD만, 즉 전신 상태 피드백 없음)해서 자연스러운
체중 이동이 저항 없이 일어나게 둔다. 스윙 다리만 스크립트로 움직인다.
착지(접촉력 검증으로 확인) 후에만 그 시점 상태를 새 평형점으로
다시 선형화해서 K를 건다.
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


def foot_fz(model, data, body_id):
    total = 0.0
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if b1 != body_id and b2 != body_id:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        frame = c.frame.reshape(3, 3)
        total += (frame.T @ force[:3])[2]
    return total


def run(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits,
        a_hip_L, a_knee_L, a_ankle_pitch_L, foot_L_body,
        lift_knee_deg, land_knee_deg, hip_target_deg,
        swing_duration_s, post_relin_interval=0.1,
        duration_s=15.0, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    phase = "stand"  # stand -> swing -> landed
    stand_s = 0.2
    swing_t0 = None
    trim_at_liftoff = None
    liftoff_t = None
    landed_t = None
    fell = None
    K, qpos_eq, qvel_eq, ctrl_trim = K0, qpos_eq0, qvel_eq0, np.zeros(model.nu)
    tick_since_relin = 0

    for tick in range(int(duration_s / 0.01)):
        fz = foot_fz(model, data, foot_L_body)

        if phase == "stand" and data.time >= stand_s:
            phase = "swing"
            swing_t0 = data.time
            trim_at_liftoff = data.ctrl.copy()
        elif phase == "swing" and liftoff_t is None and fz < 0.5:
            liftoff_t = data.time
        elif phase == "swing" and (data.time - swing_t0) >= swing_duration_s and fz > 1.0:
            phase = "landed"
            landed_t = data.time
            tick_since_relin = 0

        if phase == "landed":
            if tick_since_relin % int(post_relin_interval / 0.01) == 0:
                qpos_eq = data.qpos.copy()
                qvel_eq = np.zeros_like(qvel_eq0)
                ctrl_trim = data.ctrl.copy()
                K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
            tick_since_relin += 1

        for _ in range(2):
            if phase == "stand":
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
            elif phase == "swing":
                # 전신 피드백 끔 - 이륙 시점 trim을 그대로 유지(로컬 PD만)
                u = trim_at_liftoff.copy()
            else:  # landed
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
                dv = data.qvel - qvel_eq
                x = np.concatenate([dq, dv])
                u = ctrl_trim - K @ x

            if phase == "swing":
                s = min(1.0, (data.time - swing_t0) / swing_duration_s)
                u[a_hip_L] = math.radians(hip_target_deg) * s
                u[a_knee_L] = math.radians(lift_knee_deg) * (1 - abs(2 * s - 1)) \
                    + math.radians(land_knee_deg) * max(0.0, 2 * s - 1)
                u[a_ankle_pitch_L] = 0.0

            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        t = common.tilt_deg(data.qpos)
        if verbose and tick % 10 == 0:
            print(f"  t={data.time:.2f} phase={phase:6s} fz_L={fz:7.2f} "
                  f"tilt={t:5.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f}")
        if t > 55 and fell is None:
            fell = data.time
            break

    return liftoff_t, landed_t, fell


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
    foot_L_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")

    print("=== 스윙 중 전신 피드백 OFF (이동으로만 취급) - swing_duration 스윕 ===")
    for swing_dur in [0.3, 0.5, 0.8, 1.2, 2.0]:
        for knee_deg in [15.0, 25.0]:
            liftoff, landed, fell = run(
                model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits,
                a_hip_L, a_knee_L, a_ankle_pitch_L, foot_L_body,
                lift_knee_deg=knee_deg, land_knee_deg=0.0, hip_target_deg=15.0,
                swing_duration_s=swing_dur, duration_s=15.0)
            print(f"swing_dur={swing_dur:4.1f} knee={knee_deg:5.1f}: "
                  f"liftoff={liftoff} landed={landed} fell={fell}")

    print()
    print("=== 상세 추적: swing_dur=0.5, knee=25 ===")
    run(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits,
        a_hip_L, a_knee_L, a_ankle_pitch_L, foot_L_body,
        lift_knee_deg=25.0, land_knee_deg=0.0, hip_target_deg=15.0,
        swing_duration_s=0.5, duration_s=10.0, verbose=True)


if __name__ == "__main__":
    main()
