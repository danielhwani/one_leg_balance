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
    K = common.solve_dare(A, B, Q, R)
    return K


def main():
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    qadr_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]
    qadr_L = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_R")]

    LEAN_DEG = 20.0
    RAMP_T = 5.0
    HOLD_T = 10.0
    targets = [-LEAN_DEG, 0.0, LEAN_DEG, 0.0]

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    # K0/qpos_eq0로의 "리셋"을 하지 않는다 - 전환(램프) 중에도 마지막
    # 재선형화된 K/기준점을 그대로 들고 가고, hip_abduct만 덧붙여서
    # 민다(처음 전환만 K0 사용, 그 뒤로는 절대 K0로 안 돌아감).
    K = K0
    qpos_eq = qpos_eq0
    qvel_eq = qvel_eq0
    ctrl_trim = np.zeros(nu)

    prev_bias = 0.0
    fell = None
    for cycle in range(8):  # 2바퀴(targets 4개 x 2)
        target_bias = math.radians(targets[cycle % len(targets)])
        seg_t0 = data.time

        n_ramp = int(RAMP_T / 0.01)
        for tick in range(n_ramp):
            s = tick / n_ramp
            bias = prev_bias + (target_bias - prev_bias) * s
            hip_L_cmd = min(bias, 0.0)
            hip_R_cmd = max(bias, 0.0)
            # 램프 중인 두 관절은 기준점(qpos_eq)도 같이 움직여줘야
            # K가 "이 관절이 많이 틀어졌다"고 오해해서 큰 교정을
            #걸지 않는다 - 나머지 관절(몸통/다른 다리 등)은 그대로
            # 고정된 기준을 쓴다.
            qpos_eq_ramp = qpos_eq.copy()
            qpos_eq_ramp[qadr_R] = hip_L_cmd
            qpos_eq_ramp[qadr_L] = hip_R_cmd
            for _ in range(2):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq_ramp, data.qpos)
                dv = data.qvel - qvel_eq
                x = np.concatenate([dq, dv])
                # trim에서 hip_abduct_L/L 두 칸은 "더하기"가 아니라
                # 절대값으로 덮어써야 한다(이전 holding 구간의 값이
                # 이미 trim에 들어있어서, 더하면 이중으로 세게 됨).
                ctrl_trim_eff = ctrl_trim.copy()
                ctrl_trim_eff[a_abduct_R] = hip_L_cmd
                ctrl_trim_eff[a_abduct_L] = hip_R_cmd
                u = ctrl_trim_eff - K @ x
                u = np.clip(u, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)
            t = common.tilt_deg(data.qpos)
            if t > 55:
                fell = data.time
                break
        if fell:
            break
        prev_bias = target_bias

        qpos_eq = data.qpos.copy()
        qvel_eq = np.zeros_like(qvel_eq0)
        ctrl_trim = data.ctrl.copy()
        K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
        n_hold = int(HOLD_T / 0.01)
        for tick in range(n_hold):
            if tick % 100 == 0:
                qpos_eq = data.qpos.copy()
                ctrl_trim = data.ctrl.copy()
                K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
            for _ in range(2):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
                dv = data.qvel - qvel_eq
                x = np.concatenate([dq, dv])
                u = np.clip(ctrl_trim - K @ x, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)
            t = common.tilt_deg(data.qpos)
            if t > 55:
                fell = data.time
                break
        zx, zy = common.contact_zmp(model, data)
        print(f"cycle {cycle} (target={math.degrees(target_bias):.0f}deg) 끝: t={data.time:.2f} "
              f"zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f}")
        if fell:
            break

    if fell:
        print("FELL at t=%.2f" % fell)
    else:
        print("2바퀴(8구간) 전부 통과")


if __name__ == "__main__":
    main()
