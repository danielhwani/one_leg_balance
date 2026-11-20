import math

import numpy as np
import mujoco

import common


def main():
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A, B = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K = common.solve_dare(A, B, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")

    TARGET_DEG = -20.0
    RAMP_T = 5.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    fell = None
    for tick in range(int(15.0 / 0.01)):
        s = min(1.0, data.time / RAMP_T)
        ff = math.radians(TARGET_DEG) * s
        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = -K @ x
            u[a_abduct_R] += ff
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        if tick % 20 == 0:
            zx, zy = common.contact_zmp(model, data)
            t = common.tilt_deg(data.qpos)
            actual = math.degrees(data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]])
            print(f"t={data.time:.2f} s={s:.2f} ff={math.degrees(ff):.1f}deg actual={actual:.1f}deg zmp_y={zy:.3f} tilt={t:.2f}")
        t = common.tilt_deg(data.qpos)
        if t > 55 and fell is None:
            fell = data.time
            print("FELL at t=%.2f" % fell)
            break
    else:
        print("완료 - 안 넘어짐")


if __name__ == "__main__":
    main()
