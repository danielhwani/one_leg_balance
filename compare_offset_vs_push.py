"""compare_offset_vs_push.py
----------------------------
"외력(속도 교란)엔 잘 버티는데 초기 자세(위치) 오프셋엔 왜 못
버티나"라는 질문을 직접 비교해서 확인한다 - 작은 hip_abduct_L
오프셋부터 스윕해서, K0가 실제로 어느 크기부터 못 버티는지 보고
같은 K0의 push 내구성(이미 알고 있는 0.2m/s 생존/0.3m/s 실패)과
나란히 놓고 비교한다.
"""
import math

import numpy as np
import mujoco

import common


def run_offset(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, offset_deg, duration=30.0):
    nv = model.nv
    qadr_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qpos[qadr_R] += math.radians(offset_deg)
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None
    for tick in range(int(duration / 0.01)):
        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = np.clip(-K0 @ x, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        t = common.tilt_deg(data.qpos)
        if t > 55 and fell is None:
            fell = data.time
            break
    return fell


def run_push(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, push_mps, duration=30.0):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    data.qvel[0] += push_mps
    mujoco.mj_forward(model, data)
    fell = None
    for tick in range(int(duration / 0.01)):
        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = np.clip(-K0 @ x, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        t = common.tilt_deg(data.qpos)
        if t > 55 and fell is None:
            fell = data.time
            break
    return fell


def main():
    model = common.load_model()
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    print("=== hip_abduct_L 초기 오프셋(속도=0) 스윕 ===")
    for deg in [0.5, 1, 2, 3, 4, 5, 6, 8, 10]:
        fell = run_offset(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, deg)
        print(f"offset={deg}deg: fell={fell}")

    print()
    print("=== 전진 push(위치=평형, 속도 교란) 스윕 - 비교용 ===")
    for push in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
        fell = run_push(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, push)
        print(f"push={push}m/s: fell={fell}")


if __name__ == "__main__":
    main()
