"""view_push_demo_final.py
---------------------------
2026-11-23 최종 결론 데모: 발+골반 확대 + 다리 액추에이터 Kp/Kv
x3.0 + 타임스텝 0.0005초 + R x2 - push 복원 한계가 원래 0.25 m/s
에서 앞뒤 0.80 m/s, 좌우 1.20 m/s까지 늘어났다(README 참고).

화살표 키로 직접 밀어서 확인:
  Up/Down: 앞/뒤로 밀기 (+-X)
  Left/Right: 왼쪽/오른쪽으로 밀기 (+-Y)
한 번 누르면 0.3 m/s씩 누적 - 2~3번 눌러서 0.6~0.9 m/s 정도의
세기를 만들어보면 된다. 창을 닫으면 종료.

실행:
  python view_push_demo_final.py
"""
import math
import time

import numpy as np
import mujoco
import mujoco.viewer

import common
from pelvis_width_test import widen_pelvis
from foot_and_pelvis_test import enlarge_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv


def main():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    enlarge_feet(model, 0.24, 0.16)
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

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    PUSH_SPEED_MPS = 0.3
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
    pending_push = np.zeros(3)
    push_total = np.zeros(2)  # 누적 세기 표시용(x,y)

    def key_callback(keycode):
        if keycode == KEY_UP:
            pending_push[0] += PUSH_SPEED_MPS
            push_total[0] += PUSH_SPEED_MPS
            print(f"앞으로 밀기(+X {PUSH_SPEED_MPS} m/s, 누적 x={push_total[0]:.2f})")
        elif keycode == KEY_DOWN:
            pending_push[0] -= PUSH_SPEED_MPS
            push_total[0] -= PUSH_SPEED_MPS
            print(f"뒤로 밀기(-X {PUSH_SPEED_MPS} m/s, 누적 x={push_total[0]:.2f})")
        elif keycode == KEY_LEFT:
            pending_push[1] += PUSH_SPEED_MPS
            push_total[1] += PUSH_SPEED_MPS
            print(f"왼쪽으로 밀기(+Y {PUSH_SPEED_MPS} m/s, 누적 y={push_total[1]:.2f})")
        elif keycode == KEY_RIGHT:
            pending_push[1] -= PUSH_SPEED_MPS
            push_total[1] -= PUSH_SPEED_MPS
            print(f"오른쪽으로 밀기(-Y {PUSH_SPEED_MPS} m/s, 누적 y={push_total[1]:.2f})")

    print("발+골반 확대 + Kp/Kv x3 + 타임스텝 0.0005 + R x2 - 최종 구성")
    print("예전엔 0.25 m/s면 쓰러졌는데, 이젠 앞뒤 0.8/좌우 1.2 m/s까지 버텨야 한다.")
    print("화살표 키로 밀어보기(한 번에 0.3 m/s 누적). 창을 닫으면 종료.")

    tick = 0
    PRINT_EVERY_SEC = 0.5
    print_interval = max(1, int(PRINT_EVERY_SEC / 0.01))

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 7.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        while viewer.is_running():
            if np.any(pending_push):
                data.qvel[0:3] += pending_push
                pending_push[:] = 0.0

            for _ in range(n_sub):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = np.clip(-K0 @ x, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(max(0.0, 0.01 - 0.0005))

            tick += 1
            if tick % print_interval == 0:
                print(f"    t={data.time:6.2f}s 기울기={common.tilt_deg(data.qpos):5.2f}도 "
                      f"|v|={np.linalg.norm(data.qvel[:3]):.3f}")
            if common.tilt_deg(data.qpos) > 55:
                print(f"쓰러짐! t={data.time:.2f} (누적 push x={push_total[0]:.2f} y={push_total[1]:.2f})")
                push_total[:] = 0.0


if __name__ == "__main__":
    main()
