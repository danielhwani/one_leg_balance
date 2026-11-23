"""view_baseline_final.py
---------------------------
검증된 기반으로 복구 - 서있기(고정 K0) + 외력 복원(가슴에 실제
힘) + 좌우/앞뒤 흔들기를 하나의 데모로 통합. 오늘 최종 구조(발+
골반 확대 + Kp/Kv x3 + 타임스텝 0.0005 + R x2) 위에서 동작한다.

키:
  Up/Down/Left/Right: 가슴에 짧게(0.1초) 외력(등가 0.3 m/s씩 누적)
  L: 좌우 흔들기 모드
  F: 앞뒤 흔들기 모드
  S: 정지(흔들기 끄고 가만히 서있기) - 기본 모드
창을 닫으면 종료.

실행:
  python view_baseline_final.py
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

    chest_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chest")
    total_mass = sum(model.body_mass)
    PUSH_EQUIV_MPS = 0.3
    PUSH_DURATION_S = 0.1
    PUSH_FORCE_MAG = PUSH_EQUIV_MPS * total_mass / PUSH_DURATION_S

    AMP_LATERAL = 3.0
    AMP_FB = 4.0
    PERIOD_S = 1.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    mode = ["stand"]
    active_pushes = []  # (end_time, force_vec)

    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
    KEY_L, KEY_F, KEY_S = ord("L"), ord("F"), ord("S")

    def key_callback(keycode):
        vec = np.zeros(3)
        if keycode == KEY_UP:
            vec[0] = PUSH_FORCE_MAG
            print(f"가슴 밀기: 앞으로(등가 {PUSH_EQUIV_MPS} m/s)")
        elif keycode == KEY_DOWN:
            vec[0] = -PUSH_FORCE_MAG
            print(f"가슴 밀기: 뒤로(등가 {PUSH_EQUIV_MPS} m/s)")
        elif keycode == KEY_LEFT:
            vec[1] = PUSH_FORCE_MAG
            print(f"가슴 밀기: 로봇 왼쪽으로(등가 {PUSH_EQUIV_MPS} m/s)")
        elif keycode == KEY_RIGHT:
            vec[1] = -PUSH_FORCE_MAG
            print(f"가슴 밀기: 로봇 오른쪽으로(등가 {PUSH_EQUIV_MPS} m/s)")
        elif keycode == KEY_L:
            mode[0] = "lateral"
            print("모드: 좌우 흔들기")
            return
        elif keycode == KEY_F:
            mode[0] = "fb"
            print("모드: 앞뒤 흔들기")
            return
        elif keycode == KEY_S:
            mode[0] = "stand"
            print("모드: 정지(가만히 서있기)")
            return
        else:
            return
        active_pushes.append([data.time + PUSH_DURATION_S, vec])

    print("검증된 기반 데모: 서있기 + 외력복원(가슴) + 좌우/앞뒤 흔들기")
    print("화살표 키: 가슴에 외력 / L: 좌우 흔들기 / F: 앞뒤 흔들기 / S: 정지(기본)")
    print("창을 닫으면 종료.")

    tick = 0
    print_interval = max(1, int(0.5 / 0.01))

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 7.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        while viewer.is_running():
            active_pushes[:] = [p for p in active_pushes if p[0] > data.time]
            if active_pushes:
                data.xfrc_applied[chest_id, 0:3] = sum(p[1] for p in active_pushes)
            else:
                data.xfrc_applied[chest_id, 0:3] = 0.0

            if mode[0] == "lateral":
                bias = AMP_LATERAL * math.sin(2 * math.pi * data.time / PERIOD_S)
                bias_rad = math.radians(bias)
                sway_L, sway_R = min(bias_rad, 0.0), max(bias_rad, 0.0)
            elif mode[0] == "fb":
                bias_rad = math.radians(AMP_FB * math.sin(2 * math.pi * data.time / PERIOD_S))
                sway_L = sway_R = bias_rad
            else:
                sway_L = sway_R = None

            for _ in range(n_sub):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
                if sway_L is not None:
                    if mode[0] == "lateral":
                        u[a_abduct_L] = sway_L
                        u[a_abduct_R] = sway_R
                    else:
                        u[a_hip_L] = sway_L
                        u[a_hip_R] = sway_R
                u = np.clip(u, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(max(0.0, 0.01 - 0.0005))

            tick += 1
            if tick % print_interval == 0:
                print(f"    t={data.time:6.2f}s [{mode[0]}] 기울기={common.tilt_deg(data.qpos):5.2f}도")
            if common.tilt_deg(data.qpos) > 55:
                print(f"쓰러짐! t={data.time:.2f}")


if __name__ == "__main__":
    main()
