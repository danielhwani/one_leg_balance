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

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")
    qadr = model.jnt_qposadr[jid]

    TARGET_DEG = -20.0
    RAMP_T = 5.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    qpos_eq = qpos_eq0.copy()

    fell = None
    for tick in range(int(15.0 / 0.01)):
        s = min(1.0, data.time / RAMP_T)
        qpos_eq[qadr] = math.radians(TARGET_DEG) * s
        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = np.clip(-K @ x, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        if tick % 20 == 0:
            zx, zy = common.contact_zmp(model, data)
            t = common.tilt_deg(data.qpos)
            print(f"t={data.time:.2f} s={s:.2f} target={math.degrees(qpos_eq[qadr]):.1f}deg "
                  f"actual_hip_abduct_L={math.degrees(data.qpos[qadr]):.1f}deg zmp_y={zy:.3f} tilt={t:.2f}")
        t = common.tilt_deg(data.qpos)
        if t > 55 and fell is None:
            fell = data.time
            print("FELL at t=%.2f" % fell)
            break
    else:
        print("완료 - 안 넘어짐")


if __name__ == "__main__":
    main()
