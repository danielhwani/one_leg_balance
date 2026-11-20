"""view_landing_absorb.py
--------------------------
landing_absorb.py (swing_dur=0.8, land_knee=20)의 "넘어지는 모습"을
MuJoCo 뷰어로 직접 본다. 콘솔에 phase 전환과 tilt를 같이 찍는다.

실행:
  python view_landing_absorb.py

창을 닫으면 종료.
"""
import math
import time

import numpy as np
import mujoco
import mujoco.viewer

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


def main():
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_L")
    a_knee_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_L")
    a_ankle_pitch_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_pitch_L")
    foot_L_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")

    # landing_absorb.py 기본 파라미터 (swing_dur=0.8, land_knee=20)
    lift_knee_deg = 25.0
    land_knee_deg = 20.0
    hip_target_deg = 15.0
    swing_duration_s = 0.8
    touch_fz = 5.0
    touch_dwell_s = 0.02
    absorb_deep_deg = 35.0
    absorb_down_s = 0.12
    absorb_up_s = 0.25
    settle_v_thresh = 0.3
    post_relin_interval = 0.1
    stand_s = 0.2

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    phase = "stand"
    swing_t0 = None
    trim_at_liftoff = None
    touch_t0 = None
    absorb_t0 = None
    has_lifted = False
    knee_at_touch = 0.0
    K, qpos_eq, qvel_eq, ctrl_trim = K0, qpos_eq0, qvel_eq0, np.zeros(nu)
    tick_since_relin = 0

    print("stand -> swing -> absorb -> landed 순으로 진행. 창을 닫으면 종료.")
    tick = 0
    last_print = -1.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            fz = foot_fz(model, data, foot_L_body)
            vmag = np.linalg.norm(data.qvel[:3])

            if phase == "stand" and data.time >= stand_s:
                phase = "swing"
                swing_t0 = data.time
                trim_at_liftoff = data.ctrl.copy()
                print(f"t={data.time:.2f}: swing 시작")
            elif phase == "swing":
                if not has_lifted:
                    if fz < 1.0:
                        has_lifted = True
                elif fz > touch_fz:
                    if touch_t0 is None:
                        touch_t0 = data.time
                    elif data.time - touch_t0 >= touch_dwell_s:
                        phase = "absorb"
                        absorb_t0 = data.time
                        knee_at_touch = data.ctrl[a_knee_L]
                        print(f"t={data.time:.2f}: absorb 시작 (착지 흡수)")
                else:
                    touch_t0 = None
            elif phase == "absorb":
                absorb_total = absorb_down_s + absorb_up_s
                if (data.time - absorb_t0) >= absorb_total and vmag < settle_v_thresh:
                    phase = "landed"
                    tick_since_relin = 0
                    print(f"t={data.time:.2f}: landed 전환 (재선형화 시작)")

            if phase == "landed":
                if tick_since_relin % int(post_relin_interval / 0.01) == 0:
                    qpos_eq = data.qpos.copy()
                    qvel_eq = np.zeros_like(qvel_eq0)
                    ctrl_trim = data.ctrl.copy()
                    K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
                tick_since_relin += 1

            if phase == "stand":
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
            elif phase == "swing":
                u = trim_at_liftoff.copy()
            elif phase == "absorb":
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
            else:
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
            elif phase == "absorb":
                dt = data.time - absorb_t0
                if dt < absorb_down_s:
                    sd = dt / absorb_down_s
                    knee_at_touch_deg = math.degrees(knee_at_touch)
                    target = knee_at_touch_deg + (absorb_deep_deg - knee_at_touch_deg) * sd
                else:
                    su = min(1.0, (dt - absorb_down_s) / absorb_up_s)
                    target = absorb_deep_deg + (land_knee_deg - absorb_deep_deg) * su
                u[a_knee_L] = math.radians(target)
                u[a_ankle_pitch_L] = 0.0

            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
            viewer.sync()

            t = common.tilt_deg(data.qpos)
            if data.time - last_print >= 0.2:
                last_print = data.time
                print(f"  t={data.time:5.2f} phase={phase:7s} fz_L={fz:7.2f} "
                      f"tilt={t:5.2f} |v|={vmag:.3f}")

            tick += 1
            time.sleep(max(0.0, model.opt.timestep - 0.0005))


if __name__ == "__main__":
    main()
