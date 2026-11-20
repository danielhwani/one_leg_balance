"""quasi_static_lift.py
------------------------
어제 네 가지 시도(직접 스크립트/착지 후 재선형화/체중이동 선행/
동시진행) 전부 "발이 땅에서 떨어지는 순간" 자체에서 실패했다 -
스윙 속도(0.15~2초)와 무관하게 같은 지점에서 무너졌다.

오늘은 아예 다른 각도: 스윙을 "빠르게 지나가는 동작"이 아니라
**거의 정적으로(5~10초) 아주 천천히** 진행하면서, 계속(0.2초마다)
재선형화해 매 순간을 새 평형점으로 다시 붙잡는다. 이게 되면 "한
발 지지 평형점"이 실제로 존재한다는 증거가 되고, 안 되면 이
경로(hip_abduct_R 체중이동 + hip_L/knee_L 리프트) 자체로는 CoM이
오른발 위로 충분히 안 간다는 뜻이다.
"""
import math

import numpy as np
import mujoco

import common


def relinearize_at(model, qpos, ctrl, Q, R):
    nv, nu = model.nv, model.nu
    data_lin = mujoco.MjData(model)
    data_lin.qpos[:] = qpos
    data_lin.qvel[:] = 0.0
    data_lin.ctrl[:] = ctrl
    mujoco.mj_forward(model, data_lin)
    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data_lin, 1e-6, True, A, B, None, None)
    return common.solve_dare(A, B, Q, R)


def run(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits,
        a_abduct_R, a_hip_L, a_knee_L, a_ankle_pitch_L,
        weight_shift_deg, lift_knee_deg, hip_target_deg,
        lift_duration_s, relin_interval=0.2, duration_s=30.0, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None

    K, qpos_eq, qvel_eq, ctrl_trim = K0, qpos_eq0, qvel_eq0, np.zeros(model.nu)
    tick_since_relin = 0

    for tick in range(int(duration_s / 0.01)):
        if tick_since_relin % int(relin_interval / 0.01) == 0:
            qpos_eq = data.qpos.copy()
            qvel_eq = np.zeros_like(qvel_eq0)
            ctrl_trim = data.ctrl.copy()
            K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
        tick_since_relin += 1

        s = min(1.0, data.time / lift_duration_s)

        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
            dv = data.qvel - qvel_eq
            x = np.concatenate([dq, dv])
            u = ctrl_trim - K @ x

            u[a_abduct_R] += math.radians(weight_shift_deg) * s
            u[a_hip_L] = math.radians(hip_target_deg) * s
            u[a_knee_L] = math.radians(lift_knee_deg) * s
            u[a_ankle_pitch_L] = 0.0

            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        t = common.tilt_deg(data.qpos)
        if verbose and tick % 50 == 0:
            print(f"  t={data.time:.2f} s={s:.2f} tilt={t:.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f}")
        if t > 55 and fell is None:
            fell = data.time
            break
    return fell


def main():
    model = common.load_model()
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    a_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_L")
    a_knee_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_L")
    a_ankle_pitch_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_pitch_L")

    print("=== 무릎만 단독으로(체중이동 없음), lift_duration 스윕 ===")
    for lift_dur in [5.0, 10.0, 20.0, 40.0]:
        fell = run(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits,
                   a_abduct_R, a_hip_L, a_knee_L, a_ankle_pitch_L,
                   weight_shift_deg=0.0, lift_knee_deg=30.0, hip_target_deg=0.0,
                   lift_duration_s=lift_dur, relin_interval=0.1, duration_s=lift_dur + 20.0)
        knee_at_fail = 30.0 * fell / lift_dur if fell else None
        print(f"lift_duration={lift_dur}: fell={fell} (무릎각도 at fail={knee_at_fail})")


if __name__ == "__main__":
    main()
