"""capture_point_test.py
-------------------------
Capture Point(DCM) 개념을 이 로봇에 처음 적용해본다. 공식:
  capture_point = CoM_xy + CoM_velocity_xy / omega0
  omega0 = sqrt(g / CoM_height)
먼저 이미 검증된 push-recovery 시나리오(가슴에 실제 힘)에 적용해서,
캡처 포인트가 지지 다각형(발 영역) 경계와 비교해 "넘어질지 말지"를
의미있게 예측하는지 확인한다. 아직 스텝을 밟지는 않음 - 진단만.
"""
import math

import numpy as np
import mujoco

import common
from pelvis_width_test import widen_pelvis
from foot_offset_test import set_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv

GRAVITY = 9.81


def run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, chest_id, n_sub,
        force_vec, push_duration=0.1, duration=6.0, verbose=True):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    com_prev = data.subtree_com[1].copy()
    dt_outer = n_sub * model.opt.timestep
    fell = None

    for tick in range(int(duration / dt_outer)):
        if data.time < push_duration:
            data.xfrc_applied[chest_id, 0:3] = force_vec
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

        tilt = common.tilt_deg(data.qpos)
        if verbose and tick % 5 == 0:
            print(f"  t={data.time:5.2f} tilt={tilt:5.2f} com=({com_now[0]:+.3f},{com_now[1]:+.3f}) "
                  f"com_vel=({com_vel[0]:+.3f},{com_vel[1]:+.3f}) CP=({cp_x:+.3f},{cp_y:+.3f})")
        if tilt > 55 and fell is None:
            fell = data.time
            if verbose:
                print(f"  >>> 쓰러짐! t={fell:.2f}")
            break
    return fell


def main():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.28, 0.16, 0.0)
    scale_kp_kv(model, LEG_ACTUATORS, 3.0)
    model.opt.timestep = 0.0005
    n_sub = round(0.01 / model.opt.timestep)
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    R = R * 2.0
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    chest_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chest")
    total_mass = sum(model.body_mass)
    push_duration = 0.1

    print("지지 다각형 참고: 두 발 합쳐 x=[-0.28,0.28], y=[-0.56,0.56] (convex hull)")
    print()
    for push_mps, label in [(0.50, "성공할 것으로 예상"), (0.70, "실패할 것으로 예상")]:
        force_mag = push_mps * total_mass / push_duration
        force_vec = np.array([force_mag, 0.0, 0.0])
        print(f"=== sagittal push={push_mps} m/s ({label}) ===")
        run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, chest_id, n_sub, force_vec,
            push_duration=push_duration, duration=4.0)
        print()


if __name__ == "__main__":
    main()
