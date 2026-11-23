"""view_capture_point_push.py
------------------------------
Capture Point(DCM)를 직접 뷰어로 본다 - 가슴에 외력을 주고
(화살표 키), 매 순간 캡처 포인트를 땅에 노란 공으로 표시한다
(CoM 투영은 하늘색 공). 발 지지 경계(사지탈 ±0.28m)를 넘으면
콘솔에 경고를 찍는다. 스텝 발동은 아직 구현 안 함 - 시각 확인만.

실행:
  python view_capture_point_push.py
화살표 키로 가슴에 외력(등가 0.1 m/s씩 누적). 창을 닫으면 종료.
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

GRAVITY = 9.81
SUPPORT_X = 0.28   # 발 사지탈 반경(both feet, same x range)
SUPPORT_Y_OUT = 0.56  # 발 바깥쪽 경계(두 발 합친 convex hull)


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

    chest_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chest")
    total_mass = sum(model.body_mass)
    PUSH_EQUIV_MPS = 0.1
    PUSH_DURATION_S = 0.1
    PUSH_FORCE_MAG = PUSH_EQUIV_MPS * total_mass / PUSH_DURATION_S

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    active_pushes = []
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262

    def key_callback(keycode):
        vec = np.zeros(3)
        if keycode == KEY_UP:
            vec[0] = PUSH_FORCE_MAG
        elif keycode == KEY_DOWN:
            vec[0] = -PUSH_FORCE_MAG
        elif keycode == KEY_LEFT:
            vec[1] = PUSH_FORCE_MAG
        elif keycode == KEY_RIGHT:
            vec[1] = -PUSH_FORCE_MAG
        else:
            return
        active_pushes.append([data.time + PUSH_DURATION_S, vec])
        print(f"밀기! 누적 등가 push={sum(np.linalg.norm(p[1]) for p in active_pushes)*PUSH_DURATION_S/total_mass:.2f} m/s 상당")

    print("Capture Point 시각화: 노란 공=캡처포인트, 하늘색 공=CoM 투영")
    print("화살표 키로 밀기(한 번에 등가 0.1 m/s, 여러 번 눌러서 누적 가능)")
    print(f"지지 경계: 사지탈 x=±{SUPPORT_X}m. 넘으면 콘솔에 경고.")
    print("창을 닫으면 종료.")

    com_prev = data.subtree_com[1].copy()
    dt_outer = n_sub * model.opt.timestep
    last_print = -1.0
    warned = False

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
                u = -K0 @ x
                u = np.clip(u, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)

            com_now = data.subtree_com[1].copy()
            com_vel = (com_now - com_prev) / dt_outer
            com_prev = com_now
            z_com = com_now[2]
            omega0 = math.sqrt(GRAVITY / z_com)
            cp_x = com_now[0] + com_vel[0] / omega0
            cp_y = com_now[1] + com_vel[1] / omega0

            viewer.user_scn.ngeom = 0
            mujoco.mjv_initGeom(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                                 np.array([0.035, 0, 0]), np.array([cp_x, cp_y, 0.015]),
                                 np.eye(3).flatten(), np.array([1.0, 1.0, 0.0, 1.0]))
            mujoco.mjv_initGeom(viewer.user_scn.geoms[1], mujoco.mjtGeom.mjGEOM_SPHERE,
                                 np.array([0.03, 0, 0]), np.array([com_now[0], com_now[1], 0.015]),
                                 np.eye(3).flatten(), np.array([0.1, 0.8, 1.0, 1.0]))
            viewer.user_scn.ngeom = 2

            viewer.sync()
            time.sleep(max(0.0, 0.01 - 0.0005))

            tilt = common.tilt_deg(data.qpos)
            outside = abs(cp_x) > SUPPORT_X or abs(cp_y) > SUPPORT_Y_OUT
            if outside and not warned:
                warned = True
                print(f"  >>> CP가 지지 경계를 넘었다! t={data.time:.2f} CP=({cp_x:+.3f},{cp_y:+.3f})")
            if not outside:
                warned = False
            if data.time - last_print >= 0.3:
                last_print = data.time
                print(f"  t={data.time:6.2f}s tilt={tilt:5.2f} CP=({cp_x:+.3f},{cp_y:+.3f}) "
                      f"{'[경계 밖!]' if outside else ''}")
            if tilt > 55:
                print(f"  >>> 쓰러짐! t={data.time:.2f}")


if __name__ == "__main__":
    main()
