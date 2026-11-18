"""two_leg_baseline.py
----------------------
두 발 서있기가 이 새 프로젝트에서도 똑같이 되는지 확인하는 기준점
스크립트. computer_graphics 쪽에서 이미 300초 검증된 것과 같은
방법 - 외발 균형을 붙이기 전에 "원래 되던 것"이 깨지지 않았는지
항상 이걸로 먼저 확인한다.
"""
import math

import numpy as np
import mujoco

import common


def main():
    model = common.load_model()
    qpos_eq, qvel_eq = common.find_equilibrium(model)
    A, B = common.linearize(model, qpos_eq, qvel_eq)
    Q, R = common.build_cost_two_leg(model)
    K = common.solve_dare(A, B, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * model.nu)

    data = mujoco.MjData(model)
    nv = model.nv
    fell = None
    for tick in range(int(60.0 / 0.01)):
        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
            dv = data.qvel - qvel_eq
            x = np.concatenate([dq, dv])
            u = np.clip(-K @ x, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        t = common.tilt_deg(data.qpos)
        if t > 45 and fell is None:
            fell = data.time
            break
    print(("FELL t=%.2f" % fell) if fell else "STABLE 60s (baseline OK)")


if __name__ == "__main__":
    main()
