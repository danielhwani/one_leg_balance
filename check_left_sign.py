import math

import mujoco

import common


def main():
    model = common.load_model()
    qpos_eq, qvel_eq = common.find_equilibrium(model)
    act_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")

    for deg in [-8, -5, 5, 8]:
        data = mujoco.MjData(model)
        data.qpos[:] = qpos_eq
        data.qvel[:] = qvel_eq
        for _ in range(int(1.5 / model.opt.timestep)):
            data.ctrl[act_L] = math.radians(deg)
            mujoco.mj_step(model, data)
        zx, zy = common.contact_zmp(model, data)
        print(f"hip_abduct_R={deg:+d}deg -> zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f}")


if __name__ == "__main__":
    main()
