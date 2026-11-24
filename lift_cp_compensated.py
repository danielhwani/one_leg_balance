"""lift_cp_compensated.py
--------------------------
lift_cp_gated.py에서 발견: 스윙 다리(s2)가 어느 정도 이상
굽혀지면 그 자체가 캡처 포인트를 오히려 되돌린다(하한 위반으로
s2가 진동하며 s2~0.35~0.44에서 못 나감). 팔(shoulder_abduct_L)은
이미 액추에이터 한계(±30도)에 포화돼 있어 더 못 쓴다 - 남은
유일한 수단은 체중이동(hip_abduct_R)을 s2에 비례해서 더 키우는
것. ff_weight = WEIGHT_SHIFT_DEG + COMP_GAIN*s2 로 보상한다.
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
        a_abduct_R, a_sh_L, a_hip_R, a_knee_R, foot_red_id, foot_blue_id,
        cp_y_min, cp_y_max, s2_rate, comp_gain_deg, duration_s=60.0, verbose=False):
    nv = model.nv
    WEIGHT_SHIFT_DEG = 12.0
    SHOULDER_BASE_DEG = 45.0
    PHASE1_RAMP_T = 2.0
    PHASE1_SETTLE_T = 13.0
    PHASE1_END = PHASE1_RAMP_T + PHASE1_SETTLE_T
    LIFT_KNEE_DEG = 30.0
    LIFT_HIP_DEG = 20.0

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
        if verbose and tick % 20 == 0:
            fz_blue = foot_fz(model, data, foot_blue_id)
            print(f"  t={data.time:5.2f} tilt={tilt:6.2f} CP_y={cp_y:+.3f} s2={s2:.3f} "
                  f"ff_weight={math.degrees(ff_weight):5.1f}deg "
                  f"fz_빨강={fz_red:7.1f} fz_파랑={fz_blue:7.1f}")
        if liftoff_t is None and t > PHASE1_END and fz_red < 0.5:
            liftoff_t = data.time
            if verbose:
                print(f"  >>> 이륙! t={liftoff_t:.3f}")
        if tilt > 55 and fell_t is None:
            fell_t = data.time
            if verbose:
                print(f"  >>> 쓰러짐! t={fell_t:.3f}")
            break
    return liftoff_t, fell_t, s2


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
    foot_red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    foot_blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")

    print("=== 체중이동 보상(comp_gain_deg) 스윕, s2_rate=0.1, cp범위=[0.20,0.50] ===")
    for comp_gain in [0.0, 5.0, 10.0, 15.0, 20.0]:
        liftoff, fell, s2_final = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub,
                                       a_abduct_R, a_sh_L, a_hip_R, a_knee_R, foot_red_id, foot_blue_id,
                                       cp_y_min=0.20, cp_y_max=0.50, s2_rate=0.1,
                                       comp_gain_deg=comp_gain, duration_s=60.0,
                                       verbose=(comp_gain == 15.0))
        print(f"comp_gain={comp_gain}deg: liftoff={liftoff} fell={fell} s2_final={s2_final:.3f}")


if __name__ == "__main__":
    main()
