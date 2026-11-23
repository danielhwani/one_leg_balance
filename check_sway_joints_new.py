"""check_sway_joints_new.py
----------------------------
"움직임은 주로 어느 관절이 주도하니?" - 오늘 최종 구조(발+골반
확대 + Kp/Kv x3 + 타임스텝 0.0005 + R x2) 위에서 좌우/앞뒤 흔들기
중 각 관절이 얼마나 움직이는지 다시 분석한다. 예전(원래 구조)
결과: 좌우는 명령한 관절보다 waist_roll이 더 크게 움직이고(전신
협응 우세), 앞뒤는 명령한 hip 자체가 가장 크게 움직였다(직접
관절 우세) - 구조가 바뀐 지금도 같은지 확인.
"""
import math

import numpy as np
import mujoco

import common
from pelvis_width_test import widen_pelvis
from foot_offset_test import set_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv


def build_best_model():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.28, 0.16, 0.0)
    scale_kp_kv(model, LEG_ACTUATORS, 3.0)
    model.opt.timestep = 0.0005
    return model


def analyze(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub, mode,
            a_abduct_L=None, a_abduct_R=None, a_hip_L=None, a_hip_R=None,
            amplitude_deg=3.0, period_s=1.0, duration_s=10.0):
    nv, nu = model.nv, model.nu
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    max_dev = np.zeros(nu)
    qadr_by_act = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(nu)]

    for tick in range(int(duration_s / 0.01)):
        bias = math.radians(amplitude_deg * math.sin(2 * math.pi * data.time / period_s))
        for _ in range(n_sub):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = -K0 @ x
            if mode == "lateral":
                u[a_abduct_L] = min(bias, 0.0)
                u[a_abduct_R] = max(bias, 0.0)
            else:
                u[a_hip_L] = bias
                u[a_hip_R] = bias
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        for i in range(nu):
            dev = abs(data.qpos[qadr_by_act[i]] - qpos_eq0[qadr_by_act[i]])
            max_dev[i] = max(max_dev[i], dev)

    order = np.argsort(-max_dev)
    for i in order[:12]:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {name}: {math.degrees(max_dev[i]):.3f}도")


def main():
    model = build_best_model()
    n_sub = round(0.01 / model.opt.timestep)
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    R = R * 2.0
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    a_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_L")
    a_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_R")

    print("=== 좌우 흔들기(amp=3deg) - 관절별 최대 이탈 ===")
    analyze(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub, "lateral",
            a_abduct_L=a_abduct_L, a_abduct_R=a_abduct_R, amplitude_deg=3.0)

    print()
    print("=== 앞뒤 흔들기(amp=4deg) - 관절별 최대 이탈 ===")
    analyze(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub, "fb",
            a_hip_L=a_hip_L, a_hip_R=a_hip_R, amplitude_deg=4.0)


if __name__ == "__main__":
    main()
