"""view_sway_with_push.py
--------------------------
좌우로 계속 흔드는(view_sway_continuous.py) 도중에 화살표 키로
직접 외력(push)을 줘볼 수 있는 데모. sway_with_push.py에서 확인한
대로, 흔드는 중에는 가만히 서있을 때보다 더 센 외력(0.4m/s)까지
버틴다 - 직접 눌러서 확인해보자.

Up/Down/Left/Right: 그 방향으로 툭 밀기(여러 번 누르면 세게 누적).
창을 닫으면 종료.

실행:
  python view_sway_with_push.py
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

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    qadr_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_R")]
    qadr_L = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]

    AMPLITUDE_DEG = 3.0
    PERIOD_S = 1.0
    PUSH_SPEED_MPS = 0.3  # 사이클 내 모든 타이밍에서 안전하다고 확인된 값

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    # GLFW 키코드 - 로컬 glfw 패키지로 확인한 값(computer_graphics
    # 프로젝트의 view_humanoid_stand.py와 동일).
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
    pending_push = np.zeros(3)

    def key_callback(keycode):
        if keycode == KEY_UP:
            pending_push[0] += PUSH_SPEED_MPS
            print(f"앞으로 밀기(+X {PUSH_SPEED_MPS} m/s)")
        elif keycode == KEY_DOWN:
            pending_push[0] -= PUSH_SPEED_MPS
            print(f"뒤로 밀기(-X {PUSH_SPEED_MPS} m/s)")
        elif keycode == KEY_LEFT:
            pending_push[1] += PUSH_SPEED_MPS
            print(f"왼쪽으로 밀기(+Y {PUSH_SPEED_MPS} m/s)")
        elif keycode == KEY_RIGHT:
            pending_push[1] -= PUSH_SPEED_MPS
            print(f"오른쪽으로 밀기(-Y {PUSH_SPEED_MPS} m/s)")

    print(f"좌우로 계속 흔드는 중 (진폭 {AMPLITUDE_DEG:.0f}도, 주기 {PERIOD_S:.0f}초).")
    print("화살표 키: 그 방향으로 툭 밀기(여러 번 누르면 세게 누적). 창을 닫으면 종료.")

    tick = 0
    PRINT_EVERY_SEC = 1.0
    print_interval = max(1, int(PRINT_EVERY_SEC / 0.01))

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if np.any(pending_push):
                data.qvel[0:3] += pending_push
                pending_push[:] = 0.0

            bias = math.radians(AMPLITUDE_DEG * math.sin(2 * math.pi * data.time / PERIOD_S))
            hip_R_cmd = min(bias, 0.0)
            hip_L_cmd = max(bias, 0.0)

            for _ in range(2):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
                u[a_abduct_R] = hip_R_cmd
                u[a_abduct_L] = hip_L_cmd
                u = np.clip(u, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - 0.0005))

            tick += 1
            if tick % print_interval == 0:
                print(f"    t={data.time:6.2f}s  기울기={common.tilt_deg(data.qpos):5.2f}도  "
                      f"엉덩이R={math.degrees(data.qpos[qadr_R]):+6.2f}도  "
                      f"엉덩이L={math.degrees(data.qpos[qadr_L]):+6.2f}도")


if __name__ == "__main__":
    main()
