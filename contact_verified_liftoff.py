"""contact_verified_liftoff.py
------------------------------
사용자 제안("가만히 서있다가 한 발을 내딛으면 체중은 자연히
반대쪽으로 쏠리고, 발이 땅에 닿는 순간 진동이 생기는데 거기서
제어가 버틴다")을 검증하기 전에 두 가지를 먼저 확인한다.

1) 지금까지의 근본원인 진단(step_recovery_v1/v2)은 "착지 순간"이
   아니라 "왼발이 땅에서 '실제로' 떨어지는 순간"에 속도가 10배
   뛰며 무너졌다 - 그런데 그 실험들에서 접촉력을 직접 재서 "진짜로
   언제 뗐는지" 확인한 적이 없다. 추정(스크립트 진행률)만 썼다.

2) quasi_static_lift.py도 마찬가지 - 접촉력 체크 코드가 초안에서
   버그로 빠진 채 실행됐다. 무릎을 30도까지 구부렸을 때 왼발이
   실제로 땅에서 떨어졌는지조차 확인 안 됐다.

이 스크립트는 접촉력을 직접 측정해서(mj_contactForce, foot_R
바디 소속 geom만 필터) "왼발이 실제로 뜬 시점"을 정확히 찍고,
그 전후로 tilt/velocity가 어떻게 바뀌는지를 본다. 스윙/체중이동은
전혀 하지 않고 무릎만 다양한 속도로 살짝 구부려 "수직으로만"
든다 - 방향 이동(hip) 없이 순수하게 "접촉을 떼는 것" 자체가
K0(두 발 서있기 게인)로 버틸 수 있는 일인지만 분리해서 본다.
"""
import math

import numpy as np
import mujoco

import common


def left_foot_fz(model, data, foot_R_body):
    total = 0.0
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if b1 != foot_R_body and b2 != foot_R_body:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        frame = c.frame.reshape(3, 3)
        fz_world = (frame.T @ force[:3])[2]
        total += fz_world
    return total


def run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_knee_R, foot_R_body,
        knee_deg, lift_duration_s, duration_s=15.0, verbose=False):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    liftoff_t = None
    fell = None
    fz0 = left_foot_fz(model, data, foot_R_body)

    for tick in range(int(duration_s / 0.01)):
        s = min(1.0, data.time / lift_duration_s)
        for _ in range(2):
            dq = np.zeros(model.nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = -K0 @ x
            u[a_knee_R] = math.radians(knee_deg) * s
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        fz = left_foot_fz(model, data, foot_R_body)
        if liftoff_t is None and fz < 0.5:
            liftoff_t = data.time
        t = common.tilt_deg(data.qpos)
        vmag = np.linalg.norm(data.qvel[:3])
        if verbose and tick % 20 == 0:
            print(f"  t={data.time:.2f} s={s:.2f} fz_L={fz:7.2f} tilt={t:5.2f} |v|={vmag:.4f}")
        if t > 55 and fell is None:
            fell = data.time
            break
    return liftoff_t, fell


def main():
    model = common.load_model()
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_knee_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_R")
    foot_R_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")

    print("=== 순수 수직 들어올리기(방향 이동 없음) - 접촉력으로 실제 이륙 시점 확인 ===")
    print("    (무릎 각도, 드는 시간) -> 실제 이륙 시각 / 쓰러진 시각")
    for knee_deg in [5.0, 10.0, 15.0, 20.0, 30.0]:
        for lift_dur in [0.3, 1.0, 5.0, 20.0]:
            liftoff_t, fell = run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits,
                                   a_knee_R, foot_R_body,
                                   knee_deg=knee_deg, lift_duration_s=lift_dur,
                                   duration_s=max(15.0, lift_dur + 10.0))
            survive_after = (fell - liftoff_t) if (liftoff_t and fell) else None
            print(f"knee={knee_deg:5.1f} lift_dur={lift_dur:5.1f}: "
                  f"liftoff={liftoff_t} fell={fell} survive_after_liftoff={survive_after}")

    print()
    print("=== 상세 추적: knee=20도, lift_dur=1.0초 ===")
    run(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, a_knee_R, foot_R_body,
        knee_deg=20.0, lift_duration_s=1.0, duration_s=10.0, verbose=True)


if __name__ == "__main__":
    main()
