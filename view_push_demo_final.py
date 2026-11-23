"""view_push_demo_final.py
---------------------------
2026-11-23 최종 결론 데모: 발+골반 확대 + 다리 액추에이터 Kp/Kv
x3.0 + 타임스텝 0.0005초 + R x2 - push 복원 한계가 원래 0.25 m/s
에서 늘어났다(README 참고).

push_chest_test.py에서 확인한 대로, **가슴(chest)에 짧게(0.1초)
실제 힘(xfrc_applied)을 주는 방식**으로 테스트한다 - 단순 속도킥
(회전 없음)보다 현실적이다(회전 토크가 같이 생겨서 특히 앞뒤
방향이 더 약해진다). 등가 push_mps 기준 한계: **앞뒤 ~0.5~0.6
m/s, 좌우 ~1.05 m/s**(1.2에서 실패).

화살표 키로 직접 밀어서 확인:
  Up/Down: 앞/뒤로 밀기 (+-X)
  Left/Right: 왼쪽/오른쪽으로 밀기 (+-Y)
한 번 누르면 등가 0.3 m/s 충격(가슴에 0.1초간 실제 힘)을 준다 -
여러 번 누르면 겹쳐서 세진다. 창을 닫으면 종료.

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
from foot_offset_test import set_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv


def main():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    # 발을 발목 중심으로 대칭으로(기존엔 앞으로 0.06 치우쳐 있어서
    # 뒤로 넘어지는 쪽이 약했다: 앞 0.8/뒤 0.4 m/s 비대칭).
    # 0.30으로 하면 앞뒤 0.8/0.8까지 대칭이지만 좌우가 1.20->1.00으로
    # 줄어드는 부수효과가 있어서, 조금 줄인 0.28로 절충(앞뒤 0.75/0.75,
    # 좌우 ~1.05) - 사용자 요청.
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

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    chest_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chest")
    total_mass = sum(model.body_mass)
    PUSH_EQUIV_MPS = 0.3
    PUSH_DURATION_S = 0.1
    PUSH_FORCE_MAG = PUSH_EQUIV_MPS * total_mass / PUSH_DURATION_S

    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
    active_pushes = []  # (end_time, force_vec) 리스트 - 겹치면 합산

    def key_callback(keycode):
        vec = np.zeros(3)
        if keycode == KEY_UP:
            vec[0] = PUSH_FORCE_MAG
            print(f"앞으로 밀기(등가 {PUSH_EQUIV_MPS} m/s, 가슴에 {PUSH_FORCE_MAG:.0f}N x {PUSH_DURATION_S}s)")
        elif keycode == KEY_DOWN:
            vec[0] = -PUSH_FORCE_MAG
            print(f"뒤로 밀기(등가 {PUSH_EQUIV_MPS} m/s, 가슴에 {PUSH_FORCE_MAG:.0f}N x {PUSH_DURATION_S}s)")
        elif keycode == KEY_LEFT:
            vec[1] = PUSH_FORCE_MAG
            print(f"왼쪽으로 밀기(등가 {PUSH_EQUIV_MPS} m/s, 가슴에 {PUSH_FORCE_MAG:.0f}N x {PUSH_DURATION_S}s)")
        elif keycode == KEY_RIGHT:
            vec[1] = -PUSH_FORCE_MAG
            print(f"오른쪽으로 밀기(등가 {PUSH_EQUIV_MPS} m/s, 가슴에 {PUSH_FORCE_MAG:.0f}N x {PUSH_DURATION_S}s)")
        else:
            return
        active_pushes.append([data.time + PUSH_DURATION_S, vec])

    print("발+골반 확대(발 대칭 0.56m) + Kp/Kv x3 + 타임스텝 0.0005 + R x2 - 최종 구성")
    print("가슴에 짧게(0.1초) 실제 힘을 준다(회전 토크 포함, push_chest_test.py와 동일 모델) -")
    print("앞뒤는 등가 ~0.5~0.6 m/s, 좌우는 ~1.05 m/s까지 버텨야 한다.")
    print("화살표 키로 밀어보기(한 번에 등가 0.3 m/s, 여러 번 누르면 겹쳐서 세짐). 창을 닫으면 종료.")

    tick = 0
    PRINT_EVERY_SEC = 0.5
    print_interval = max(1, int(PRINT_EVERY_SEC / 0.01))

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
                print(f"쓰러짐! t={data.time:.2f}")


if __name__ == "__main__":
    main()
