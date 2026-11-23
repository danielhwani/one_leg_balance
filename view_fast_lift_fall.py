"""view_fast_lift_fall.py
--------------------------
가만히 서있는 상태에서(체중이동 없이) 빨간발(foot_R)을 빠르게
들어올려서 쓰러지는 과정을 직접 본다. 고정 K0는 이런 큰 전이를
못 버틴다는 걸 보여주는 데모 - 쓰러지는 게 정상/예상된 결과.

실행:
  python view_fast_lift_fall.py
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

    LIFT_KNEE_DEG = 30.0
    LIFT_HIP_DEG = 20.0
    LIFT_RAMP_T = 0.3   # 빠르게

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    print(f"체중이동 없이, 빨간발을 {LIFT_RAMP_T}초 만에 빠르게 들어올린다 - 쓰러지는 게 정상.")
    print("창을 닫으면 종료.")

    last_print = -1.0
    liftoff_t = None
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 7.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        while viewer.is_running():
            t = data.time
            s = min(1.0, t / LIFT_RAMP_T)
            ff_knee = math.radians(LIFT_KNEE_DEG) * s
            ff_hip = math.radians(LIFT_HIP_DEG) * s

            for _ in range(n_sub):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
                u[a_knee_R] += ff_knee
                u[a_hip_R] += ff_hip
                u = np.clip(u, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(max(0.0, 0.01 - 0.0005))

            tilt = common.tilt_deg(data.qpos)
            fz_red = foot_fz(model, data, foot_red_id)
            if liftoff_t is None and fz_red < 0.5:
                liftoff_t = data.time
                print(f">>> 빨간발 실제 이륙! t={liftoff_t:.3f}")
            if data.time - last_print >= 0.1:
                last_print = data.time
                fz_blue = foot_fz(model, data, foot_blue_id)
                print(f"  t={data.time:5.2f}s tilt={tilt:6.2f} fz_빨강={fz_red:7.2f} fz_파랑={fz_blue:7.2f}")
            if tilt > 55:
                print(f">>> 쓰러짐! t={data.time:.3f}")


if __name__ == "__main__":
    main()
