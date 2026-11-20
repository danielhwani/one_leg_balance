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
    qadr_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]

    def run(target_deg, ramp_t, total_t=25.0, verbose=False):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos_eq0
        data.qvel[:] = qvel_eq0
        fell = None
        for tick in range(int(total_t / 0.01)):
            s = min(1.0, data.time / ramp_t)
            ff = math.radians(target_deg) * s
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
            t = common.tilt_deg(data.qpos)
            if t > 55 and fell is None:
                fell = data.time
                break
            if verbose and tick % 100 == 0:
                zx, zy = common.contact_zmp(model, data)
                actual = math.degrees(data.qpos[qadr_R])
                print(f"  t={data.time:.2f} ff={math.degrees(ff):.0f}deg actual={actual:.1f}deg zmp_y={zy:.3f} tilt={t:.2f}")
        zx, zy = common.contact_zmp(model, data)
        actual = math.degrees(data.qpos[qadr_R])
        return fell, actual, zy, common.tilt_deg(data.qpos)

    print("=== 피드포워드 세기 스윕 (ramp 5초, 총 25초) ===")
    for target_deg in [-20, -40, -60, -90, -120, -150, -180]:
        fell, actual, zy, t = run(target_deg, ramp_t=5.0)
        print(f"target={target_deg}deg: fell={fell} actual={actual:.1f}deg zmp_y={zy:.3f} tilt={t:.2f}")

    print()
    print("=== -90deg 자세히 ===")
    run(-90, ramp_t=5.0, verbose=True)


if __name__ == "__main__":
    main()
