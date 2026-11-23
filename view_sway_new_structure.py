"""view_sway_new_structure.py
------------------------------
외력이 아니라 스스로 좌우/앞뒤로 계속 흔드는 것(sway_continuous.py
/ sway_forward_back.py 방식)을 오늘 최종 구성(발+골반 확대 +
Kp/Kv x3 + 타임스텝 0.0005 + R x2) 위에서 본다.

키:
  L: 좌우 흔들기 모드
  F: 앞뒤 흔들기 모드
  S: 정지(흔들기 끄고 가만히 서있기)
창을 닫으면 종료.

실행:
  python view_sway_new_structure.py
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

    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    a_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_L")
    a_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_R")

    AMP_LATERAL = 3.0
    AMP_FB = 4.0
    PERIOD_S = 1.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    mode = ["lateral"]
    KEY_L, KEY_F, KEY_S = ord("L"), ord("F"), ord("S")

    def key_callback(keycode):
        if keycode == KEY_L:
            mode[0] = "lateral"
            print("모드: 좌우 흔들기")
        elif keycode == KEY_F:
            mode[0] = "fb"
            print("모드: 앞뒤 흔들기")
        elif keycode == KEY_S:
            mode[0] = "stand"
            print("모드: 정지(가만히 서있기)")

    print("발+골반 확대 + Kp/Kv x3 + 타임스텝 0.0005 + R x2 위에서 흔들기 데모")
    print("L: 좌우 흔들기 / F: 앞뒤 흔들기 / S: 정지. 창을 닫으면 종료.")

    tick = 0
    print_interval = max(1, int(0.5 / 0.01))

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 7.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        while viewer.is_running():
            if mode[0] == "lateral":
                bias = AMP_LATERAL * math.sin(2 * math.pi * data.time / PERIOD_S)
                bias_rad = math.radians(bias)
                hip_abduct_L_cmd = min(bias_rad, 0.0)
                hip_abduct_R_cmd = max(bias_rad, 0.0)
                hip_L_cmd = hip_R_cmd = None
            elif mode[0] == "fb":
                bias_rad = math.radians(AMP_FB * math.sin(2 * math.pi * data.time / PERIOD_S))
                hip_L_cmd = hip_R_cmd = bias_rad
                hip_abduct_L_cmd = hip_abduct_R_cmd = None
            else:
                hip_abduct_L_cmd = hip_abduct_R_cmd = hip_L_cmd = hip_R_cmd = None

            for _ in range(n_sub):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
                if hip_abduct_L_cmd is not None:
                    u[a_abduct_L] = hip_abduct_L_cmd
                    u[a_abduct_R] = hip_abduct_R_cmd
                if hip_L_cmd is not None:
                    u[a_hip_L] = hip_L_cmd
                    u[a_hip_R] = hip_R_cmd
                u = np.clip(u, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(max(0.0, 0.01 - 0.0005))

            tick += 1
            if tick % print_interval == 0:
                print(f"    t={data.time:6.2f}s [{mode[0]}] 기울기={common.tilt_deg(data.qpos):5.2f}도")


if __name__ == "__main__":
    main()
