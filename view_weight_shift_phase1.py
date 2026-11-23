"""view_weight_shift_phase1.py
-------------------------------
"오른발(빨간) 내딛기" 중 1단계만: 체중을 파란발(foot_L)로 옮기기.
간접 명령 방식 - 빨간 다리(hip_abduct_R)에 양수 각도를 명령하면
따로 체중이동 신호 없이도 파란발에 자동으로, 깨끗하게 체중이
쏠린다(정강이-땅 접촉 버그 수정 후 검증: 목표 12도까지 60초 생존
+ 완전 수렴 - fz_파랑=171.2N, fz_빨강=75.1N, tilt=1.30도로 고정).

주의: 이 양수 명령은 빨간발을 "벌리는" 게 아니라 오히려 살짝
안쪽으로 당긴다(부수효과) - 2단계(빨간발을 실제로 들어서 바깥으로
내딛기)는 아직 미해결, 여기서는 1단계(체중이동)만 시연한다.

실행:
  python view_weight_shift_phase1.py
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

    TARGET_DEG = 12.0
    RAMP_T = 2.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    print("1단계: 체중을 파란발(foot_L)로 이동 - 빨간 다리 고관절(hip_abduct_R)")
    print(f"에 간접 명령(목표 {TARGET_DEG}도, {RAMP_T:.0f}초 램프).")
    print("t=2~3초면 체중의 대부분(~63%)이 이미 파란발로 옮겨간다 - 완전")
    print("수렴(34초)까지는 느린 꼬리가 더 남지만 위험하지 않음.")
    print("빨간발은 벌어지는 게 아니라 살짝 안쪽으로 당겨지는 부수효과만 있음.")
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
                print(f"  t={data.time:6.2f}s tilt={tilt:5.2f} fz_빨강={fz_red:6.1f} fz_파랑={fz_blue:6.1f}")


if __name__ == "__main__":
    main()
