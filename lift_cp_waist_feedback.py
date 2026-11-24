"""lift_cp_waist_feedback.py
-----------------------------
lift_cp_compensated.py에서 발견: 이륙 전 준비(CP 게이트+체중이동
보상)를 아무리 잘해도, 실제 이륙 후에는 항상 ~0.5초 안에 쓰러진다.
이번엔 "이륙 후" 구간에 허리(waist_roll)로 CP 피드백을 추가해서
그 생존시간 자체를 늘릴 수 있는지 테스트한다 - 팔은 이미
액추에이터 한계에 포화돼 못 쓰므로, 범위가 더 큰(±25도) 허리를
쓴다.
"""
import math

import numpy as np
import mujoco

import common
from pelvis_width_test import widen_pelvis
from foot_offset_test import set_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv

GRAVITY = 9.81


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


def run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub,
        a_abduct_R, a_sh_L, a_hip_R, a_knee_R, a_waist_roll, foot_red_id, foot_blue_id,
        cp_y_min, cp_y_max, s2_rate, comp_gain_deg, kp_waist, cp_y_target_post,
        duration_s=40.0, verbose=False):
    nv = model.nv
    WEIGHT_SHIFT_DEG = 12.0
    SHOULDER_BASE_DEG = 45.0
    PHASE1_RAMP_T = 2.0
    PHASE1_SETTLE_T = 13.0
    PHASE1_END = PHASE1_RAMP_T + PHASE1_SETTLE_T
    LIFT_KNEE_DEG = 30.0
    LIFT_HIP_DEG = 20.0
    WAIST_LIMIT = math.radians(24.0)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    com_prev = data.subtree_com[1].copy()
    dt_outer = n_sub * model.opt.timestep

    cp_y = 0.0
    s2 = 0.0
    liftoff_t = None
    fell_t = None
    for tick in range(int(duration_s / 0.01)):
        t = data.time
        s1 = min(1.0, t / PHASE1_RAMP_T)
        ff_arm = math.radians(SHOULDER_BASE_DEG) * s1

        if t >= PHASE1_END:
            if cp_y_min <= cp_y <= cp_y_max:
                s2 = min(1.0, s2 + s2_rate * 0.01)
            else:
                s2 = max(0.0, s2 - s2_rate * 0.01)
        ff_weight = math.radians(WEIGHT_SHIFT_DEG * s1 + comp_gain_deg * s2)
        ff_knee = math.radians(LIFT_KNEE_DEG) * s2
        ff_hip = math.radians(LIFT_HIP_DEG) * s2

        ff_waist = 0.0
        if liftoff_t is not None:
            cp_error = cp_y_target_post - cp_y
            ff_waist = kp_waist * cp_error
            ff_waist = max(-WAIST_LIMIT, min(WAIST_LIMIT, ff_waist))

        for _ in range(n_sub):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = -K0 @ x
            u[a_abduct_R] += ff_weight
            u[a_sh_L] += ff_arm
            u[a_knee_R] += ff_knee
            u[a_hip_R] += ff_hip
            u[a_waist_roll] += ff_waist
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        com_now = data.subtree_com[1].copy()
        com_vel = (com_now - com_prev) / dt_outer
        com_prev = com_now
        omega0 = math.sqrt(GRAVITY / com_now[2])
        cp_y = com_now[1] + com_vel[1] / omega0

        tilt = common.tilt_deg(data.qpos)
        fz_red = foot_fz(model, data, foot_red_id)
        if verbose and tick % 10 == 0:
            fz_blue = foot_fz(model, data, foot_blue_id)
            print(f"  t={data.time:5.2f} tilt={tilt:6.2f} CP_y={cp_y:+.3f} s2={s2:.3f} "
                  f"ff_waist={math.degrees(ff_waist):5.1f}deg fz_빨강={fz_red:7.1f} fz_파랑={fz_blue:7.1f}")
        if liftoff_t is None and t > PHASE1_END and fz_red < 0.5:
            liftoff_t = data.time
            if verbose:
                print(f"  >>> 이륙! t={liftoff_t:.3f}")
        if tilt > 55 and fell_t is None:
            fell_t = data.time
            if verbose:
                print(f"  >>> 쓰러짐! t={fell_t:.3f}")
            break
    return liftoff_t, fell_t


def main():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.28, 0.16, 0.0)
    scale_kp_kv(model, LEG_ACTUATORS, 3.0)
    model.opt.timestep = 0.0005
    n_sub = round(0.01 / model.opt.timestep)
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    R = R * 2.0
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    a_sh_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_shoulder_abduct_L")
    a_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_R")
    a_knee_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_R")
    a_waist_roll = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_waist_roll")
    foot_red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    foot_blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")

    print("=== 이륙 후 허리 CP 피드백(kp_waist) 스윕 ===")
    for kp_waist in [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]:
        liftoff, fell = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub,
                             a_abduct_R, a_sh_L, a_hip_R, a_knee_R, a_waist_roll,
                             foot_red_id, foot_blue_id,
                             cp_y_min=0.20, cp_y_max=0.50, s2_rate=0.1, comp_gain_deg=2.0,
                             kp_waist=kp_waist, cp_y_target_post=0.30,
                             duration_s=40.0, verbose=(kp_waist == 4.0))
        post = None if (liftoff is None or fell is None) else fell - liftoff
        print(f"kp_waist={kp_waist}: liftoff={liftoff} fell={fell} post_survival={post}")


if __name__ == "__main__":
    main()
