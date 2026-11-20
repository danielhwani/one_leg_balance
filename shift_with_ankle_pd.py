import math

import numpy as np
import mujoco

import common


def main():
    model = common.load_model()
    qpos_eq, qvel_eq = common.find_equilibrium(model)
    nu = model.nu

    def act(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    a_abduct_R = act("act_hip_abduct_L")
    a_ankle_roll_L = act("act_ankle_roll_L")
    a_ankle_roll_R = act("act_ankle_roll_R")
    a_ankle_pitch_L = act("act_ankle_pitch_L")
    a_ankle_pitch_R = act("act_ankle_pitch_R")

    TARGET_LEAN_DEG = 0.0
    RAMP_T = 3.0
    KP_ROLL, KD_ROLL = 15.0, 1.5
    KP_PITCH, KD_PITCH = 15.0, 1.5

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq
    data.qvel[:] = qvel_eq

    for tick in range(int(20.0 / 0.01)):
        s = min(1.0, data.time / RAMP_T)
        for _ in range(2):
            roll_err = 2.0 * data.qpos[4]
            pitch_err = 2.0 * data.qpos[5]
            roll_rate = data.qvel[3]
            pitch_rate = data.qvel[4]

            ankle_roll_fb = -KP_ROLL * roll_err - KD_ROLL * roll_rate
            ankle_pitch_fb = -KP_PITCH * pitch_err - KD_PITCH * pitch_rate

            u = np.zeros(nu)
            u[a_abduct_R] = math.radians(TARGET_LEAN_DEG) * s
            u[a_ankle_roll_L] = np.clip(ankle_roll_fb, -0.35, 0.35)
            u[a_ankle_roll_R] = np.clip(ankle_roll_fb, -0.35, 0.35)
            u[a_ankle_pitch_L] = np.clip(ankle_pitch_fb, -0.43, 0.43)
            u[a_ankle_pitch_R] = np.clip(ankle_pitch_fb, -0.43, 0.43)

            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        if tick % 20 == 0:
            zx, zy = common.contact_zmp(model, data)
            t = common.tilt_deg(data.qpos)
            v = np.linalg.norm(data.qvel[:3])
            print(f"t={data.time:.2f} s={s:.2f} zmp_y={zy:.3f} tilt={t:.2f} |v|={v:.4f}")
        if common.tilt_deg(data.qpos) > 55:
            print("FELL at t=%.2f" % data.time)
            break
    else:
        print("완료 - 안 넘어짐")


if __name__ == "__main__":
    main()
