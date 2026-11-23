"""view_abduct_right_only.py
-----------------------------
새로 확인된 메커니즘: 오른쪽 다리만 벌리면(hip_abduct_R) 따로
체중이동 신호를 주지 않아도 왼발(foot_L)에 자동으로 체중이
쏠리고, 진짜로 수렴한다(정강이-땅 접촉 버그 수정 후 검증:
목표 12도까지 60초 생존 + 완전 수렴, 16도부터 무너짐).

이 데모는 목표 12도, 5초 램프 후 유지. 빨간 다리(foot_R)가
벌어지고, 파란 다리(foot_L)가 체중을 받는 걸 색으로 확인한다.

실행:
  python view_abduct_right_only.py
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
    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")

    # 좌우 혼동 방지 - foot_R(벌어지는, 빨간) / foot_L(체중을 받는, 파란)
    for bname, rgba in [("leg_R", [0.9, 0.1, 0.1, 1.0]), ("shin_R", [0.9, 0.1, 0.1, 1.0]),
                         ("foot_R", [0.9, 0.1, 0.1, 1.0]), ("leg_L", [0.1, 0.3, 0.9, 1.0]),
                         ("shin_L", [0.1, 0.3, 0.9, 1.0]), ("foot_L", [0.1, 0.3, 0.9, 1.0])]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        for gi in range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]):
            model.geom_rgba[gi] = rgba

    TARGET_DEG = 12.0
    RAMP_T = 5.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    print(f"빨간 다리(foot_R)만 벌림 (목표 {TARGET_DEG}도, 5초 램프) - 60초 검증된 안정값")
    print("파란 다리(foot_L)에 체중이 자동으로 쏠려야 한다 (별도 체중이동 신호 없음)")
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
                fz_L = foot_fz(model, data, foot_L_id)
                fz_R = foot_fz(model, data, foot_R_id)
                tilt = common.tilt_deg(data.qpos)
                print(f"  t={data.time:6.2f}s tilt={tilt:5.2f} fz_L={fz_L:6.1f} fz_R={fz_R:6.1f}")


if __name__ == "__main__":
    main()
