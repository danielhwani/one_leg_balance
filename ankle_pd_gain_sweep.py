import math

import numpy as np
import mujoco

import common


def act(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def run(model, qpos_eq, qvel_eq, kp, kd, duration=20.0):
    nu = model.nu
    a_ankle_roll_R = act(model, "act_ankle_roll_R")
    a_ankle_roll_L = act(model, "act_ankle_roll_L")
    a_ankle_pitch_R = act(model, "act_ankle_pitch_R")
    a_ankle_pitch_L = act(model, "act_ankle_pitch_L")

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq
    data.qvel[:] = qvel_eq
    fell = None
    for tick in range(int(duration / 0.01)):
        for _ in range(2):
            roll_err = 2.0 * data.qpos[4]
            pitch_err = 2.0 * data.qpos[5]
            roll_rate = data.qvel[3]
            pitch_rate = data.qvel[4]
            ankle_roll_fb = -kp * roll_err - kd * roll_rate
            ankle_pitch_fb = -kp * pitch_err - kd * pitch_rate

            u = np.zeros(nu)
            u[a_ankle_roll_R] = np.clip(ankle_roll_fb, -0.35, 0.35)
            u[a_ankle_roll_L] = np.clip(ankle_roll_fb, -0.35, 0.35)
            u[a_ankle_pitch_R] = np.clip(ankle_pitch_fb, -0.43, 0.43)
            u[a_ankle_pitch_L] = np.clip(ankle_pitch_fb, -0.43, 0.43)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        t = common.tilt_deg(data.qpos)
        if t > 55 and fell is None:
            fell = data.time
            break
    return fell


def main():
    model = common.load_model()
    qpos_eq, qvel_eq = common.find_equilibrium(model)

    for kp, kd in [(15, 1.5), (8, 1.0), (4, 0.5), (2, 0.3), (1, 0.15), (0.5, 0.08), (0.2, 0.03)]:
        fell = run(model, qpos_eq, qvel_eq, kp, kd)
        print(f"kp={kp} kd={kd}: fell={fell}")


if __name__ == "__main__":
    main()
