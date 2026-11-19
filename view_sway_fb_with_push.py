"""view_sway_fb_with_push.py
-----------------------------
앞뒤로 계속 흔드는(view_sway_forward_back.py) 도중에 화살표 키로
외력(push)을 줘볼 수 있는 데모. sway_fb_with_push.py에서 확인한
대로, 앞뒤 흔들기는 좌우 흔들기보다 외력에 더 약하고(0.25m/s까지만
항상 안전, 0.3m/s부터 일부 타이밍에서 실패) 타이밍 의존성도 있다
- 그래서 기본 push 세기를 좌우 버전(0.3m/s)보다 더 보수적으로
(0.2m/s) 잡았다.

Up/Down/Left/Right: 그 방향으로 툭 밀기(여러 번 누르면 세게 누적).
창을 닫으면 종료.

실행:
  python view_sway_fb_with_push.py
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

    AMPLITUDE_DEG = 4.0
    PERIOD_S = 1.0
    # 앞뒤 흔들기는 좌우보다 push에 더 약하고(0.25m/s까지만 항상
    # 안전) 타이밍 의존성도 있어서, 좌우 버전(0.3m/s)보다 더
    # 보수적으로 잡았다(README의 "앞뒤 흔들기 + 외력" 섹션 참고).
    PUSH_SPEED_MPS = 0.2

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

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

    print(f"앞뒤로 계속 흔드는 중 (진폭 {AMPLITUDE_DEG:.0f}도, 주기 {PERIOD_S:.0f}초).")
    print("화살표 키: 그 방향으로 툭 밀기(여러 번 누르면 세게 누적 - 앞뒤 흔들기는 "
          "좌우보다 약하니 너무 여러 번 누르면 쓰러질 수 있음). 창을 닫으면 종료.")

    tick = 0
    PRINT_EVERY_SEC = 1.0
    print_interval = max(1, int(PRINT_EVERY_SEC / 0.01))

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if np.any(pending_push):
                data.qvel[0:3] += pending_push
                pending_push[:] = 0.0

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
