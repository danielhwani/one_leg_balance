import math

import numpy as np
import mujoco

import common


def main():
    model = common.load_model()
    qpos_eq, qvel_eq = common.find_equilibrium(model)
    act_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")

    foot_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    data0 = mujoco.MjData(model)
    data0.qpos[:] = qpos_eq
    mujoco.mj_forward(model, data0)
    print("foot_R y 범위 참고:", data0.xpos[foot_R][1] - 0.10, "~", data0.xpos[foot_R][1] + 0.10)
    print()

    for deg in [-5, -5.5, -6, -6.5, -7, -7.5, -8]:
        data = mujoco.MjData(model)
        data.qpos[:] = qpos_eq
        data.qvel[:] = qvel_eq
        for _ in range(int(2.0 / model.opt.timestep)):
            data.ctrl[act_R] = math.radians(deg)
            mujoco.mj_step(model, data)
        zx, zy = common.contact_zmp(model, data)
        t = common.tilt_deg(data.qpos)
        v = np.linalg.norm(data.qvel[:3])
        print(f"angle={deg:+.1f}deg (2초 유지) -> zmp_y={zy:.3f} tilt={t:.3f} |v|={v:.4f}")


if __name__ == "__main__":
    main()
