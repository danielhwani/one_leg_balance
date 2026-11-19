"""view_sway_continuous.py
---------------------------
같은 자리에서 계속 좌우로 살살 흔들리는 데모(사인파) - MuJoCo
뷰어 + 터미널 출력 동시. ROS2 없이 순수 터미널 + MuJoCo 창만 쓴다.

`view_sway_demo.py`(기울고-멈추고-돌아오기)보다 훨씬 간단하다 -
hip_abduct 목표를 사인파로 부드럽게 좌우로 계속 오가게 하는데,
진폭(4도)이 K0의 실제 유효 범위(±3~4도) 안에 계속 머물러서,
**재선형화도 복귀 단계도 전혀 필요 없이 처음부터 끝까지 원래
K0 하나로** 계속 흔든다(`sway_continuous.py`로 150초 검증 완료).

실행:
  python view_sway_continuous.py
"""
import math
import time

import numpy as np
import mujoco
import mujoco.viewer

import common


def main():
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    qadr_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_R")]
    qadr_L = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]

    AMPLITUDE_DEG = 4.0  # K0 범위(±3~4도) 안에서만
    PERIOD_S = 6.0  # 한 바퀴(오른쪽->왼쪽->오른쪽) 걸리는 시간

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    print(f"같은 자리에서 계속 좌우로 살살 흔들기 (진폭 {AMPLITUDE_DEG:.0f}도, "
          f"주기 {PERIOD_S:.0f}초). 창을 닫으면 종료.")

    tick = 0
    PRINT_EVERY_SEC = 1.0
    print_interval = max(1, int(PRINT_EVERY_SEC / 0.01))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            bias_deg = AMPLITUDE_DEG * math.sin(2 * math.pi * data.time / PERIOD_S)
            bias = math.radians(bias_deg)
            hip_R_cmd = min(bias, 0.0)
            hip_L_cmd = max(bias, 0.0)

            for _ in range(2):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
                u[a_abduct_R] = hip_R_cmd
                u[a_abduct_L] = hip_L_cmd
                u = np.clip(u, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - 0.0005))

            tick += 1
            if tick % print_interval == 0:
                print(f"    t={data.time:6.2f}s  기울기={common.tilt_deg(data.qpos):5.2f}도  "
                      f"엉덩이R={math.degrees(data.qpos[qadr_R]):+6.2f}도  "
                      f"엉덩이L={math.degrees(data.qpos[qadr_L]):+6.2f}도")


if __name__ == "__main__":
    main()
