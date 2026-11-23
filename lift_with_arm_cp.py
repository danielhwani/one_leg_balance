"""lift_with_arm_cp.py
-----------------------
캡처 포인트가 파란발 영역 안에 들어오도록(체중이동 12° + 왼팔
벌리기 30~60도) 준비한 다음, 빨간발을 실제로 들어올려본다.
CP가 처음으로 올바른 위치에 있는 상태에서 들어올리면 결과가
달라지는지 확인 - 쓰러져도 상관없이 전체 과정을 추적한다.
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


def main():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.28, 0.16, 0.0)
    scale_kp_kv(model, LEG_ACTUATORS, 3.0)
    model.opt.timestep = 0.0005
    n_sub = round(0.01 / model.opt.timestep)
    nv, nu = model.nv, model.nu
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

    WEIGHT_SHIFT_DEG = 12.0
    SHOULDER_DEG = 45.0
    PHASE1_RAMP_T = 2.0
    PHASE1_SETTLE_T = 13.0  # CP_y가 0.24를 확실히 넘도록 충분히 대기(검증: t=15s에 0.246)
    PHASE1_END = PHASE1_RAMP_T + PHASE1_SETTLE_T

    LIFT_KNEE_DEG = 30.0
    LIFT_HIP_DEG = 20.0
    LIFT_RAMP_T = 1.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    com_prev = data.subtree_com[1].copy()
    dt_outer = n_sub * model.opt.timestep

    liftoff_t = None
    fell_t = None
    print("=== phase1(체중이동+팔) -> phase2(빨간발 들어올리기), CP 추적 ===")
    for tick in range(int(25.0 / 0.01)):
        t = data.time
        s1 = min(1.0, t / PHASE1_RAMP_T)
        ff_weight = math.radians(WEIGHT_SHIFT_DEG) * s1
        ff_arm = math.radians(SHOULDER_DEG) * s1

        if t < PHASE1_END:
            ff_knee, ff_hip = 0.0, 0.0
        else:
            s2 = min(1.0, (t - PHASE1_END) / LIFT_RAMP_T)
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
        if tick % 20 == 0:
            fz_blue = foot_fz(model, data, foot_blue_id)
            print(f"  t={data.time:5.2f} tilt={tilt:6.2f} CP_y={cp_y:+.3f} "
                  f"fz_빨강={fz_red:7.2f} fz_파랑={fz_blue:7.2f}")
        if liftoff_t is None and data.time > PHASE1_END and fz_red < 0.5:
            liftoff_t = data.time
            print(f"  >>> 빨간발 실제 이륙! t={liftoff_t:.3f}")
        if tilt > 55 and fell_t is None:
            fell_t = data.time
            print(f"  >>> 쓰러짐! t={fell_t:.3f}")
            break

    if fell_t is None:
        print("10초 동안 버팀!")
    print(f"\n이륙시점: {liftoff_t}, 쓰러진 시점: {fell_t}")


if __name__ == "__main__":
    main()
