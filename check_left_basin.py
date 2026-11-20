import math

import numpy as np
import mujoco

import common


def run_offset(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, qadr, offset_deg, duration=30.0):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qpos[qadr] += math.radians(offset_deg)
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


def main():
    model = common.load_model()
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)
    qadr_L = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_R")]

    print("=== hip_abduct_R 초기 오프셋(양수 방향) 스윕 ===")
    for deg in [1, 2, 3, 4, 5, 6]:
        fell = run_offset(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, qadr_L, deg)
        print(f"offset=+{deg}deg: fell={fell}")


if __name__ == "__main__":
    main()
