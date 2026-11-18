import mujoco
import common


def main():
    model = common.load_model()
    qpos_eq, qvel_eq = common.find_equilibrium(model)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq
    data.qvel[:] = qvel_eq
    for tick in range(int(20.0 / 0.01)):
        for _ in range(2):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
        if tick % 50 == 0:
            t = common.tilt_deg(data.qpos)
            print(f"t={data.time:.2f} tilt={t:.4f}")
        if common.tilt_deg(data.qpos) > 55:
            print("FELL at t=%.2f" % data.time)
            break
    else:
        print("완료 - 안 넘어짐")


if __name__ == "__main__":
    main()
