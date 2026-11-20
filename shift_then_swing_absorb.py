"""shift_then_swing_absorb.py
------------------------------
사용자가 뷰어로 직접 보고 알려준 것: "발도 못 떼고 무릎이 풀리면서
뒤로 쓰러진다" - 왼발이 아직 체중을 다 받고 있는 상태에서 무릎을
구부리라고 명령하니, position-PD가 그 하중을 버틸 힘이 없어 그냥
접혀(collapse)버린 것. 체중을 오른다리로 먼저 옮겨 왼발을 비워야
무릎을 구부렸을 때 실제로 발이 뜬다 (원본 README 교훈 #5:
"체중을 한쪽으로 옮기려면 스탠스 다리만 조정해야 한다").

0) shift: view_weight_shift.py의 검증된 패턴(K0 + hip_abduct_L
   additive feedforward)으로 체중을 오른다리로 옮기고 settle.
1) swing: 그 다음에야 왼다리(hip_R/knee_R)를 스크립트로 스윙 -
   이제 왼발 하중이 거의 없으니 무릎을 구부리면 실제로 들릴 것.
2) absorb/landed: landing_absorb.py와 동일.
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
        a_abduct_R, a_hip_R, a_knee_R, a_ankle_pitch_R, foot_R_body,
        shift_deg, shift_ramp_s, shift_settle_s,
        lift_knee_deg, land_knee_deg, hip_target_deg, swing_duration_s,
        touch_fz=5.0, touch_dwell_s=0.02,
        absorb_deep_deg=35.0, absorb_down_s=0.12, absorb_up_s=0.25,
        settle_v_thresh=0.3, post_relin_interval=0.1,
        duration_s=20.0, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    phase = "shift"  # shift -> swing -> absorb -> landed
    swing_t0 = None
    trim_at_liftoff = None
    touch_t0 = None
    absorb_t0 = None
    has_lifted = False
    knee_at_touch = 0.0
    K, qpos_eq, qvel_eq, ctrl_trim = K0, qpos_eq0, qvel_eq0, np.zeros(model.nu)
    tick_since_relin = 0
    fell = None
    liftoff_t = None
    landed_t = None

    for tick in range(int(duration_s / 0.01)):
        fz = foot_fz(model, data, foot_R_body)
        vmag = np.linalg.norm(data.qvel[:3])

        if phase == "shift" and data.time >= shift_ramp_s + shift_settle_s:
            phase = "swing"
            swing_t0 = data.time
            trim_at_liftoff = data.ctrl.copy()
        elif phase == "swing":
            if not has_lifted:
                if fz < 1.0:
                    has_lifted = True
                    liftoff_t = data.time
            elif fz > touch_fz:
                if touch_t0 is None:
                    touch_t0 = data.time
                elif data.time - touch_t0 >= touch_dwell_s:
                    phase = "absorb"
                    absorb_t0 = data.time
                    knee_at_touch = data.ctrl[a_knee_R]
            else:
                touch_t0 = None
        elif phase == "absorb":
            absorb_total = absorb_down_s + absorb_up_s
            if (data.time - absorb_t0) >= absorb_total and vmag < settle_v_thresh:
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
            if phase == "shift":
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
                s_shift = min(1.0, data.time / shift_ramp_s)
                u[a_abduct_R] += math.radians(shift_deg) * s_shift
            elif phase == "swing":
                u = trim_at_liftoff.copy()
            elif phase == "absorb":
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
                u[a_abduct_R] += math.radians(shift_deg)
            else:  # landed
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
                dv = data.qvel - qvel_eq
                x = np.concatenate([dq, dv])
                u = ctrl_trim - K @ x

            if phase == "swing":
                s = min(1.0, (data.time - swing_t0) / swing_duration_s)
                u[a_hip_R] = math.radians(hip_target_deg) * s
                u[a_knee_R] = math.radians(lift_knee_deg) * (1 - abs(2 * s - 1)) \
                    + math.radians(land_knee_deg) * max(0.0, 2 * s - 1)
                u[a_ankle_pitch_R] = 0.0
            elif phase == "absorb":
                dt = data.time - absorb_t0
                if dt < absorb_down_s:
                    sd = dt / absorb_down_s
                    knee_at_touch_deg = math.degrees(knee_at_touch)
                    target = knee_at_touch_deg + (absorb_deep_deg - knee_at_touch_deg) * sd
                else:
                    su = min(1.0, (dt - absorb_down_s) / absorb_up_s)
                    target = absorb_deep_deg + (land_knee_deg - absorb_deep_deg) * su
                u[a_knee_R] = math.radians(target)
                u[a_ankle_pitch_R] = 0.0

            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        t = common.tilt_deg(data.qpos)
        if verbose and tick % 10 == 0:
            print(f"  t={data.time:.2f} phase={phase:7s} fz_L={fz:7.2f} "
                  f"tilt={t:5.2f} |v|={vmag:.4f}")
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

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_R")
    a_knee_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_R")
    a_ankle_pitch_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_pitch_R")
    foot_R_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")

    print("=== 체중이동(-20도) 먼저, 그 다음 스윙 - swing_duration 스윕 ===")
    for swing_dur in [0.3, 0.5, 0.8, 1.2]:
        liftoff, landed, fell = run(
            model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits,
            a_abduct_R, a_hip_R, a_knee_R, a_ankle_pitch_R, foot_R_body,
            shift_deg=-20.0, shift_ramp_s=5.0, shift_settle_s=5.0,
            lift_knee_deg=25.0, land_knee_deg=20.0, hip_target_deg=15.0,
            swing_duration_s=swing_dur, duration_s=22.0)
        print(f"swing_dur={swing_dur:4.1f}: liftoff={liftoff} landed={landed} fell={fell}")

    print()
    print("=== 상세 추적: swing_dur=0.5 ===")
    run(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits,
        a_abduct_R, a_hip_R, a_knee_R, a_ankle_pitch_R, foot_R_body,
        shift_deg=-20.0, shift_ramp_s=5.0, shift_settle_s=5.0,
        lift_knee_deg=25.0, land_knee_deg=20.0, hip_target_deg=15.0,
        swing_duration_s=0.5, duration_s=14.0, verbose=True)


if __name__ == "__main__":
    main()
