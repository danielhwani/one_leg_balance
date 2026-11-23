"""crouch_test.py
------------------
사용자 제안: "무게중심도 좀 낮추어 보자." 로봇 질량 분포를 바꾸는
대신(더 큰 구조 변경), 가장 간단한 방법 - **무릎을 살짝 구부린
자세(크라우치)**로 선 채로 평형점을 다시 찾는다. 사람도 스키/스케이트
처럼 안정성이 중요할 때 무릎을 굽혀서 무게중심을 낮춘다.

common.find_equilibrium은 ctrl=0(똑바로 선 자세)으로 settle한다 -
이번엔 무릎 목표를 미리 넣고 settle시켜서 크라우치 평형점을 만든다.
지금까지 최선인 발+골반 확대(0.40m 폭, 0.24x0.16 발) 위에 추가로
테스트.
"""
import math

import numpy as np
import mujoco

import common
from push_tvlqr import run_push_fixed_K0, run_push_tvlqr
from pelvis_width_test import widen_pelvis
from foot_and_pelvis_test import enlarge_feet


def find_crouch_equilibrium(model, knee_deg, ramp_s=4.0, settle_s=3.0):
    """무릎만 굽히면(고관절/발목 보정 없이) 앞으로 쓰러진다 - K0
    피드백을 켠 채로 천천히 램프시켜서(view_weight_shift.py와 같은
    레시피) 전신이 자연스럽게 균형을 다시 잡게 한다."""
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_knee_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_L")
    a_knee_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_R")

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    total_t = ramp_s + settle_s
    for _ in range(int(total_t / model.opt.timestep)):
        s = min(1.0, data.time / ramp_s)
        ff = math.radians(knee_deg) * s
        dq = np.zeros(nv)
        mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
        dv = data.qvel - qvel_eq0
        x = np.concatenate([dq, dv])
        u = -K0 @ x
        u[a_knee_L] += ff
        u[a_knee_R] += ff
        u = np.clip(u, -ctrl_limits, ctrl_limits)
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
    return data.qpos.copy(), data.qvel.copy()


def test_one_config(label, knee_deg, push_list, axis):
    model = common.load_model()
    widen_pelvis(model, 0.40)
    enlarge_feet(model, 0.24, 0.16)
    nu = model.nu

    if knee_deg > 0:
        qpos_eq0, qvel_eq0 = find_crouch_equilibrium(model, knee_deg)
    else:
        qpos_eq0, qvel_eq0 = common.find_equilibrium(model)

    pelvis_z = qpos_eq0[2]
    tilt = common.tilt_deg(qpos_eq0)
    vmag = np.linalg.norm(qvel_eq0)
    print(f"--- {label}: pelvis_z={pelvis_z:.3f} tilt={tilt:.2f} |v|={vmag:.4f} ---")
    if vmag > 0.05 or tilt > 10:
        print("    (경고: settle이 안정적으로 안 됨 - 이 크라우치 각도는 너무 급함)")

    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    for push_mps in push_list:
        fell_fixed = run_push_fixed_K0(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, push_mps, axis=axis)
        fell_tv = run_push_tvlqr(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits, push_mps, axis=axis)
        print(f"  push={push_mps:.2f}: 고정K0={'생존' if fell_fixed is None else f'fell@{fell_fixed:.2f}'}  "
              f"TV-LQR={'생존' if fell_tv is None else f'fell@{fell_tv:.2f}'}")


def main():
    pushes = [0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    for axis in [0, 1]:
        print(f"========== axis={axis} ({'sagittal' if axis == 0 else 'lateral'}) ==========")
        test_one_config("크라우치 없음(발+골반만)", 0, pushes, axis=axis)
        test_one_config("무릎 15도 크라우치", 15, pushes, axis=axis)
        test_one_config("무릎 25도 크라우치", 25, pushes, axis=axis)


if __name__ == "__main__":
    main()
