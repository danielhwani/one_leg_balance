"""view_settle_wide_stance.py
------------------------------
settle_wide_stance.py(부호 수정판: 왼발 +4도 체중이동 + 오른발 +6도
벌리기, 둘 다 5초 동안 동시에 램프)를 뷰어로 직접 본다. 29.47초에
결국 무너지는 걸 볼 수 있다 - "짧게 보면 안정적이다가 결국 터지는"
패턴.

실행:
  python view_settle_wide_stance.py
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
    a_ankle_roll_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_roll_R")
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")

    THETA_R_DEG = 6.0
    WEIGHT_SHIFT_L_DEG = 4.0  # 부호 수정됨 - 왼발로 체중이동
    RAMP_T = 5.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    # 좌우 혼동 방지 - 코드상 foot_R(벌리며 슬라이딩하는 다리)을
    # 빨간색, foot_L(체중이 쏠리는 다리)을 파란색으로.
    for bname, rgba in [("leg_R", [0.9, 0.1, 0.1, 1.0]), ("shin_R", [0.9, 0.1, 0.1, 1.0]),
                         ("foot_R", [0.9, 0.1, 0.1, 1.0]), ("leg_L", [0.1, 0.3, 0.9, 1.0]),
                         ("shin_L", [0.1, 0.3, 0.9, 1.0]), ("foot_L", [0.1, 0.3, 0.9, 1.0])]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        for gi in range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]):
            model.geom_rgba[gi] = rgba

    print("고정 K0 + 5초 램프(파란 다리=foot_L에 체중이동 +4도, 빨간 다리=foot_R 벌리기 +6도, 동시 진행)")
    print("빨간 다리가 슬라이딩하며 벌어지고, 파란 다리가 체중을 더 받아야 한다.")
    print("예전 결과: 29.47초에 결국 무너짐. 창을 닫으면 종료.")

    last_print = -1.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 7.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        while viewer.is_running():
            s = min(1.0, data.time / RAMP_T)
            ff_R = math.radians(THETA_R_DEG) * s
            ff_ankle = -ff_R
            ff_L = math.radians(WEIGHT_SHIFT_L_DEG) * s

            for _ in range(n_sub):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
                u[a_abduct_R] += ff_R
                u[a_ankle_roll_R] += ff_ankle
                u[a_abduct_L] += ff_L
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
            if common.tilt_deg(data.qpos) > 55:
                print(f"쓰러짐! t={data.time:.2f}")


if __name__ == "__main__":
    main()
