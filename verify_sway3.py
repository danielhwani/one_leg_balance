"""verify_sway3.py
-------------------
사용자 아이디어로 재구성: "기울고 유지"에만 재선형화(K_lean)를 쓰고,
중앙으로 돌아올 때는 재선형화된 K를 안 들고 가고 **곧장 원래
두 발 서있기 K0로 전환**한다 - 중앙은 K0 자신의 평형점이라 재선형화가
필요 없다. 매 "바깥쪽" 램프도 항상 K0/qpos_eq0에서 새로 시작한다
(cycle 0가 잘 됐던 이유 그대로 매번 재현) - 그래서 오차가 누적되지
않는다.

순서: [center(K0)] -> ramp(K0+override) -> presettle(K0+override 고정)
      -> hold(재선형화된 K_lean) -> ramp back(K0+override, 0으로)
      -> [center(K0), override 없음] -> 반복(반대쪽)
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

    LEAN_DEG = 20.0
    RAMP_T = 5.0
    PRE_SETTLE_T = 5.0
    HOLD_T = 10.0
    RELIN_INTERVAL = 1.0
    N_SWAYS = 4  # 오른쪽, 왼쪽, 오른쪽, 왼쪽

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    fell = None

    def step_with(K, qpos_eq, qvel_eq, ctrl_trim_base, hip_L_cmd, hip_R_cmd):
        nonlocal fell
        ctrl_trim_eff = ctrl_trim_base.copy()
        ctrl_trim_eff[a_abduct_R] = hip_L_cmd
        ctrl_trim_eff[a_abduct_L] = hip_R_cmd
        for _ in range(2):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
            dv = data.qvel - qvel_eq
            x = np.concatenate([dq, dv])
            u = np.clip(ctrl_trim_eff - K @ x, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        if common.tilt_deg(data.qpos) > 55 and fell is None:
            fell = data.time

    for sway in range(N_SWAYS):
        sign = -1.0 if sway % 2 == 0 else 1.0  # 오른쪽(-) / 왼쪽(+) 번갈아
        target_bias = sign * math.radians(LEAN_DEG)

        # 1) 바깥쪽 램프: 항상 K0/qpos_eq0에서 "새로" 시작(누적 없음).
        n_ramp = int(RAMP_T / 0.01)
        for tick in range(n_ramp):
            bias = target_bias * (tick / n_ramp)
            step_with(K0, qpos_eq0, qvel_eq0, np.zeros(nu), min(bias, 0.0), max(bias, 0.0))
            if fell:
                break
        if fell:
            break

        # 2) 정착(같은 K0, 목표에 고정) - 속도가 가라앉을 시간.
        n_settle = int(PRE_SETTLE_T / 0.01)
        for tick in range(n_settle):
            step_with(K0, qpos_eq0, qvel_eq0, np.zeros(nu), min(target_bias, 0.0), max(target_bias, 0.0))
            if fell:
                break
        if fell:
            break

        zx, zy = common.contact_zmp(model, data)
        print(f"sway {sway} 바깥쪽 끝: t={data.time:.2f} zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f}")

        # 3) 유지: 재선형화된 K_lean으로 그 자세를 버틴다.
        qpos_eq = data.qpos.copy()
        qvel_eq = np.zeros_like(qvel_eq0)
        ctrl_trim = data.ctrl.copy()
        K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
        n_hold = int(HOLD_T / 0.01)
        for tick in range(n_hold):
            if tick % int(RELIN_INTERVAL / 0.01) == 0:
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
            if common.tilt_deg(data.qpos) > 55 and fell is None:
                fell = data.time
                break
        if fell:
            break

        zx, zy = common.contact_zmp(model, data)
        actual_trim = data.ctrl.copy()
        print(f"sway {sway} 유지 끝: t={data.time:.2f} zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f}")

        # 4) 복귀 램프: K0로 전환하되, "기준점 자체"도 지금 자세에서
        #    qpos_eq0까지 서서히 옮겨가며 섞는다 - 그래야 K0가 보는
        #    오차(x)가 갑자기 커지지 않고 램프 내내 작게 유지된다
        #    (갑자기 통째로 전환하면 K0가 큰 오차를 보고 충격을 줌 -
        #    오늘 찾은 버그).
        start_R = actual_trim[a_abduct_R]
        start_L = actual_trim[a_abduct_L]
        qpos_lean = data.qpos.copy()
        for tick in range(n_ramp):
            s = tick / n_ramp
            hip_L_cmd = start_R * (1 - s)
            hip_R_cmd = start_L * (1 - s)
            qpos_eq_blend = qpos_lean * (1 - s) + qpos_eq0 * s
            step_with(K0, qpos_eq_blend, qvel_eq0, np.zeros(nu), hip_L_cmd, hip_R_cmd)
            if tick % 50 == 0:
                zx, zy = common.contact_zmp(model, data)
                print(f"    [복귀ramp] t={data.time:.2f} s={s:.2f} tilt={common.tilt_deg(data.qpos):.2f} "
                      f"zmp_y={zy:.3f} |v|={np.linalg.norm(data.qvel[:3]):.4f}")
            if fell:
                break
        if fell:
            break

        # 5) 중앙 정착: 완전히 원래 K0만으로(override 없음) 5초 유지
        #    - 이게 되는지가 "진짜로 중앙에 안전하게 돌아왔는지"의 증거.
        for tick in range(n_settle):
            step_with(K0, qpos_eq0, qvel_eq0, np.zeros(nu), 0.0, 0.0)
            if fell:
                break
        if fell:
            break

        zx, zy = common.contact_zmp(model, data)
        print(f"sway {sway} 중앙 복귀 끝: t={data.time:.2f} zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f} "
              f"|v|={np.linalg.norm(data.qvel[:3]):.4f}")

    if fell:
        print("FELL at t=%.2f" % fell)
    else:
        print(f"{N_SWAYS}번 좌우 교대 전부 통과")


if __name__ == "__main__":
    main()
