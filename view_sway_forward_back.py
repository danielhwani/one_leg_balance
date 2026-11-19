"""view_sway_forward_back.py
-----------------------------
좌우 흔들기(view_sway_continuous.py)와 같은 방식, 앞뒤로 숙이는
버전. hip_L/hip_R(피치, 양쪽 다리를 같은 방향으로)에 사인파
override를 주고, 나머지는 전부 원래 K0 하나로(재선형화 없음).

실행:
  python view_sway_forward_back.py
"""
import math
import time

import numpy as np
import mujoco
import mujoco.viewer

import common


def main():
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_L")
    a_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_R")
    qadr_hip_L = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_L")]

    AMPLITUDE_DEG = 4.0  # K0 범위(+6도/-5도, check_pitch_basin.py) 안에서만
    PERIOD_S = 1.0  # 걸음걸이 수준(150초 검증됨)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    print(f"같은 자리에서 계속 앞뒤로 살살 흔들기 (진폭 {AMPLITUDE_DEG:.0f}도, "
          f"주기 {PERIOD_S:.0f}초). 창을 닫으면 종료.")

    tick = 0
    PRINT_EVERY_SEC = 1.0
    print_interval = max(1, int(PRINT_EVERY_SEC / 0.01))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            bias = math.radians(AMPLITUDE_DEG * math.sin(2 * math.pi * data.time / PERIOD_S))

            for _ in range(2):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
                u[a_hip_L] = bias
                u[a_hip_R] = bias
                u = np.clip(u, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - 0.0005))

            tick += 1
            if tick % print_interval == 0:
                print(f"    t={data.time:6.2f}s  기울기={common.tilt_deg(data.qpos):5.2f}도  "
                      f"엉덩이피치={math.degrees(data.qpos[qadr_hip_L]):+6.2f}도")


if __name__ == "__main__":
    main()
