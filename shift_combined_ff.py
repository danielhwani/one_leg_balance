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

    def act(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    a_abduct_L = act("act_hip_abduct_L")
    a_ankle_roll_L = act("act_ankle_roll_L")
    a_ankle_roll_R = act("act_ankle_roll_R")
    a_abduct_R = act("act_hip_abduct_R")

    def run(hip_deg, ankle_deg, ramp_t, total_t=20.0, verbose=False):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos_eq0
        data.qvel[:] = qvel_eq0
        fell = None
        for tick in range(int(total_t / 0.01)):
            s = min(1.0, data.time / ramp_t)
            ff_hip = math.radians(hip_deg) * s
            ff_ankle = math.radians(ankle_deg) * s
            for _ in range(2):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K @ x
                u[a_abduct_L] += ff_hip
                u[a_ankle_roll_L] += ff_ankle
                u = np.clip(u, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)
            t = common.tilt_deg(data.qpos)
            if t > 55 and fell is None:
                fell = data.time
                break
            if verbose and tick % 50 == 0:
                zx, zy = common.contact_zmp(model, data)
                print(f"  t={data.time:.2f} s={s:.2f} zmp_y={zy:.3f} tilt={t:.2f}")
        zx, zy = common.contact_zmp(model, data)
        return fell, zy, common.tilt_deg(data.qpos)

    print("=== hip_abduct_L/L + ankle_roll_L/L 같이 강제 ===")
    for hip_deg, ankle_deg in [(-15, -10), (-15, -15), (-20, -15), (-15, -18), (-10, -18), (-18, -18)]:
        fell, zy, t = run(hip_deg, ankle_deg, ramp_t=5.0)
        print(f"hip={hip_deg} ankle={ankle_deg}: fell={fell} zmp_y={zy:.3f} tilt={t:.2f}")

    print()
    print("=== 제일 나은 조합 자세히 ===")
    run(-15, -18, ramp_t=5.0, verbose=True)


if __name__ == "__main__":
    main()
