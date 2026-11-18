"""calibrate_weight_shift.py
-----------------------------
스탠스(지지) 다리의 hip_abduct 각도를 천천히(한 값당 1초씩 유지)
바꿔가며, 그 각도에서 정적으로 안정됐을 때 ZMP_y가 어디로 가는지
곡선으로 뽑는다. README의 교훈 #5/#6 - 급하게 피드백을 돌리면
전이 현상 때문에 반대 방향으로 튀니, 먼저 "각도 -> 정적 ZMP" 관계를
깨끗하게 확보해두고 그 위에 경로를 설계한다.

양쪽 다리 다 테스트 - hip_abduct_L과 hip_abduct_R이 실제로 대칭인지도
확인(모델 정의만 보면 대칭이어야 하지만 직접 검증하는 게 안전).
"""
import math

import numpy as np
import mujoco

import common


def calibrate(model, qpos_eq, qvel_eq, abduct_act_name, angles_deg, settle_sec=1.5):
    act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, abduct_act_name)
    results = []
    for deg in angles_deg:
        data = mujoco.MjData(model)
        data.qpos[:] = qpos_eq
        data.qvel[:] = qvel_eq
        for _ in range(int(settle_sec / model.opt.timestep)):
            data.ctrl[act] = math.radians(deg)
            mujoco.mj_step(model, data)
        zx, zy = common.contact_zmp(model, data)
        t = common.tilt_deg(data.qpos)
        results.append((deg, zx, zy, t, data.qvel.copy()[:3]))
    return results


def main():
    model = common.load_model()
    qpos_eq, qvel_eq = common.find_equilibrium(model)

    angles = [0, -3, -5, -8, -10, -12, -15, -18, -20]

    print("=== hip_abduct_R (오른다리 스탠스) 캘리브레이션 ===")
    for deg, zx, zy, t, v in calibrate(model, qpos_eq, qvel_eq, "act_hip_abduct_R", angles):
        print(f"  angle={deg:+.0f}deg -> zmp=({zx:.3f},{zy:.3f}) tilt={t:.2f} |v|={np.linalg.norm(v):.4f}")

    print()
    print("=== hip_abduct_L (왼다리 스탠스) 캘리브레이션 ===")
    for deg, zx, zy, t, v in calibrate(model, qpos_eq, qvel_eq, "act_hip_abduct_L", angles):
        print(f"  angle={deg:+.0f}deg -> zmp=({zx:.3f},{zy:.3f}) tilt={t:.2f} |v|={np.linalg.norm(v):.4f}")


if __name__ == "__main__":
    main()
