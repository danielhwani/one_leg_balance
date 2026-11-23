"""push_tvlqr.py
-----------------
"서있는 과정에 외력에 반응해 균형을 유지하는 것부터" - 고정 K0 대신
TV-LQR(연속 재선형화, continuous_relinearize3.py와 같은 레시피)을
push 복구에 적용해서, 고정 K0의 기존 한계(0.25~0.4 m/s)보다 더 큰
외력도 버틸 수 있는지 확인한다.

주의(교훈): 푸시 직후의 과도 상태(transient) 동안 바로 재선형화를
시작하면 그 과도 속도를 "의도된 것"으로 받아들여 바로 불안정해진다
(continuous_relinearize 초기 버전의 교훈) - 푸시 후 DELAY초만큼
순수 K0로 버티게 한 다음에야 재선형화를 시작한다.
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


def run_push_fixed_K0(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, push_mps, axis=0, duration=15.0):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    data.qvel[axis] += push_mps
    mujoco.mj_forward(model, data)
    fell = None
    for tick in range(int(duration / 0.01)):
        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = np.clip(-K0 @ x, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        t = common.tilt_deg(data.qpos)
        if t > 55 and fell is None:
            fell = data.time
            break
    return fell


def run_push_tvlqr(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits, push_mps,
                    axis=0, delay_s=0.3, relin_interval=0.2, duration=15.0, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    data.qvel[axis] += push_mps
    mujoco.mj_forward(model, data)

    qpos_eq, qvel_eq, K, ctrl_trim = qpos_eq0, qvel_eq0, K0, np.zeros(model.nu)
    relin_started = False
    tick_since_relin = 0
    fell = None
    for tick in range(int(duration / 0.01)):
        if data.time > delay_s:
            if not relin_started:
                relin_started = True
                tick_since_relin = 0
            if tick_since_relin % int(relin_interval / 0.01) == 0:
                qpos_eq = data.qpos.copy()
                qvel_eq = np.zeros_like(qvel_eq0)
                ctrl_trim = data.ctrl.copy()
                K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
            tick_since_relin += 1

        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
            dv = data.qvel - qvel_eq
            x = np.concatenate([dq, dv])
            if relin_started:
                u = ctrl_trim - K @ x
            else:
                u = -K0 @ x
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        t = common.tilt_deg(data.qpos)
        if verbose and tick % 20 == 0:
            print(f"    t={data.time:.2f} tilt={t:.2f} |v|={np.linalg.norm(data.qvel[:3]):.3f}")
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

    print("=== 고정 K0 vs TV-LQR(연속 재선형화) push 내구성 비교 (축0=x, 전후) ===")
    for push_mps in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]:
        fell_fixed = run_push_fixed_K0(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, push_mps, axis=0)
        fell_tv = run_push_tvlqr(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits, push_mps, axis=0)
        print(f"push={push_mps:.2f} m/s: 고정K0={'생존' if fell_fixed is None else f'fell@{fell_fixed:.2f}'}  "
              f"TV-LQR={'생존' if fell_tv is None else f'fell@{fell_tv:.2f}'}")

    print()
    print("=== 상세 추적: push=0.5 m/s, TV-LQR ===")
    run_push_tvlqr(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits, 0.5, axis=0, verbose=True, duration=8.0)


if __name__ == "__main__":
    main()
