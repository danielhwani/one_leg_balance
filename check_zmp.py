"""check_zmp.py
---------------
common.contact_zmp()가 정지 서있기에서 두 발 중앙(0 근처)을 정확히
가리키는지 확인하는 기준점 스크립트.
"""
import mujoco

import common


def main():
    model = common.load_model()
    data = mujoco.MjData(model)
    qpos_eq, qvel_eq = common.find_equilibrium(model)
    data.qpos[:] = qpos_eq
    data.qvel[:] = qvel_eq
    mujoco.mj_forward(model, data)
    for tick in range(50):
        for _ in range(2):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
        if tick % 10 == 0:
            zx, zy = common.contact_zmp(model, data)
            print(f"t={data.time:.2f} zmp=({zx:.4f},{zy:.4f}) ncon={data.ncon}")


if __name__ == "__main__":
    main()
