"""view_lift_with_arm_cp.py
----------------------------
체중이동(hip_abduct_R 12도) + 왼팔 벌리기(shoulder_abduct_L 45도)로
캡처 포인트를 파란발 지지 영역(y>=0.24) 안에 15초 동안 충분히
들여놓은 다음, 빨간발을 들어올린다. 캡처 포인트(노란 공)와 CoM
투영(하늘색 공)을 실시간으로 땅에 표시한다.

검증된 결과: CP가 사전에 올바른 위치(0.246)에 있어도 들어올리는
동안 다시 흐트러져서 t=16.88s 이륙, t=17.38s 쓰러짐(이륙 후
0.5초) - 사전 위치만으로는 부족하다는 걸 보여주는 데모.

실행:
  python view_lift_with_arm_cp.py
창을 닫으면 종료.
"""
import math
import time

import numpy as np
import mujoco
import mujoco.viewer

import common
from pelvis_width_test import widen_pelvis
from foot_offset_test import set_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv

GRAVITY = 9.81
SUPPORT_BLUE_Y_MIN = 0.24


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

    for bname, rgba in [("leg_R", [0.9, 0.1, 0.1, 1.0]), ("shin_R", [0.9, 0.1, 0.1, 1.0]),
                         ("foot_R", [0.9, 0.1, 0.1, 1.0]), ("leg_L", [0.1, 0.3, 0.9, 1.0]),
                         ("shin_L", [0.1, 0.3, 0.9, 1.0]), ("foot_L", [0.1, 0.3, 0.9, 1.0])]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        for gi in range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]):
            model.geom_rgba[gi] = rgba

    WEIGHT_SHIFT_DEG = 12.0
    SHOULDER_DEG = 45.0
    PHASE1_RAMP_T = 2.0
    PHASE1_SETTLE_T = 13.0
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

    print(f"체중이동+왼팔 벌리기로 CP를 파란발 영역에 {PHASE1_END:.0f}초 준비 후 빨간발 들어올리기")
    print("노란 공=캡처포인트, 하늘색 공=CoM 투영. 창을 닫으면 종료.")

    last_print = -1.0
    liftoff_t = None
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 7.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        while viewer.is_running():
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
            cp_x = com_now[0] + com_vel[0] / omega0
            cp_y = com_now[1] + com_vel[1] / omega0

            viewer.user_scn.ngeom = 0
            mujoco.mjv_initGeom(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                                 np.array([0.035, 0, 0]), np.array([cp_x, cp_y, 0.015]),
                                 np.eye(3).flatten(), np.array([1.0, 1.0, 0.0, 1.0]))
            mujoco.mjv_initGeom(viewer.user_scn.geoms[1], mujoco.mjtGeom.mjGEOM_SPHERE,
                                 np.array([0.03, 0, 0]), np.array([com_now[0], com_now[1], 0.015]),
                                 np.eye(3).flatten(), np.array([0.1, 0.8, 1.0, 1.0]))
            viewer.user_scn.ngeom = 2

            viewer.sync()
            time.sleep(max(0.0, 0.01 - 0.0005))

            tilt = common.tilt_deg(data.qpos)
            fz_red = foot_fz(model, data, foot_red_id)
            if liftoff_t is None and data.time > PHASE1_END and fz_red < 0.5:
                liftoff_t = data.time
                print(f">>> 빨간발 실제 이륙! t={liftoff_t:.3f}")
            if data.time - last_print >= 0.3:
                last_print = data.time
                fz_blue = foot_fz(model, data, foot_blue_id)
                print(f"  t={data.time:6.2f}s tilt={tilt:5.2f} CP_y={cp_y:+.3f} "
                      f"fz_빨강={fz_red:7.1f} fz_파랑={fz_blue:7.1f}")
            if tilt > 55:
                print(f">>> 쓰러짐! t={data.time:.3f}")


if __name__ == "__main__":
    main()
