"""view_abduct_red_negative.py
-------------------------------
빨간 다리(foot_R)를 진짜로 바깥쪽으로 벌리는(음수 각도) 데모.
-2~-12도 전부 60초 수렴 검증됨 - 목표 -8도로 시연.

색: 빨간 다리(foot_R)=벌어지는 다리, 파란 다리(foot_L)=그대로.
확인된 바: 벌어지는 빨간 다리가 오히려 체중을 더 받는다
(fz_빨강 > fz_파랑), 양수(좁히는 방향)일 때와는 반대.

실행:
  python view_abduct_red_negative.py
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

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    foot_red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    foot_blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")

    for bname, rgba in [("leg_R", [0.9, 0.1, 0.1, 1.0]), ("shin_R", [0.9, 0.1, 0.1, 1.0]),
                         ("foot_R", [0.9, 0.1, 0.1, 1.0]), ("leg_L", [0.1, 0.3, 0.9, 1.0]),
                         ("shin_L", [0.1, 0.3, 0.9, 1.0]), ("foot_L", [0.1, 0.3, 0.9, 1.0])]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        for gi in range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]):
            model.geom_rgba[gi] = rgba

    TARGET_DEG = -8.0
    RAMP_T = 5.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    print(f"빨간 다리(foot_R) 진짜로 바깥 벌리기 (목표 {TARGET_DEG}도, 5초 램프)")
    print("60초 검증된 안정값 - 벌어지는 빨간 다리가 체중을 더 받아야 한다")
    print("창을 닫으면 종료.")

    last_print = -1.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 7.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        while viewer.is_running():
            s = min(1.0, data.time / RAMP_T)
            ff_R = math.radians(TARGET_DEG) * s

            for _ in range(n_sub):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
                u[a_abduct_R] += ff_R
                u = np.clip(u, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(max(0.0, 0.01 - 0.0005))

            if data.time - last_print >= 0.5:
                last_print = data.time
                fz_red = foot_fz(model, data, foot_red_id)
                fz_blue = foot_fz(model, data, foot_blue_id)
                tilt = common.tilt_deg(data.qpos)
                red_y = data.xpos[foot_red_id][1]
                print(f"  t={data.time:6.2f}s tilt={tilt:5.2f} fz_빨강={fz_red:6.1f} fz_파랑={fz_blue:6.1f} "
                      f"빨강발_y={red_y:.4f}")


if __name__ == "__main__":
    main()
