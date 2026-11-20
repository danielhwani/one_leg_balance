"""verify_sway2.py
-------------------
verify_sway.py의 버그 수정판: 램프가 끝난 직후 바로 재선형화하지
않고, README 교훈 5번대로 "램프 -> (같은 K로) 5초 정착 -> 그제서야
재선형화 시작"을 매 구간(오른쪽 기울기, 복귀, 왼쪽 기울기, 복귀...)
마다 똑같이 적용한다. 복귀 램프도 "하나의 새로운 목표를 향한 전환"
으로 취급 - K0로 리셋하지 않고 마지막으로 재선형화된 K를 그대로
들고 간다.
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

    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    qadr_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]
    qadr_L = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_R")]

    LEAN_DEG = 20.0
    RAMP_T = 5.0
    PRE_SETTLE_T = 5.0
    HOLD_T = 10.0
    RELIN_INTERVAL = 1.0
    targets = [-LEAN_DEG, 0.0, LEAN_DEG, 0.0]

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    # 전환(램프+정착)에 쓰는 K/기준점 - 절대 K0로 "리셋"하지 않고,
    # 매번 마지막 hold에서 재선형화된 값을 그대로 이어받는다.
    K = K0
    qpos_eq = qpos_eq0.copy()
    qvel_eq = qvel_eq0.copy()
    ctrl_trim = np.zeros(nu)

    prev_bias = 0.0
    fell = None

    def run_segment(seconds, bias_fn, verbose_tag, relin_during=False):
        nonlocal fell, K, qpos_eq, qvel_eq, ctrl_trim
        print(f"  >>> {verbose_tag} 시작: ctrl_trim[R]={math.degrees(ctrl_trim[a_abduct_L]):.2f}deg "
              f"K_norm={np.linalg.norm(K):.3f} qpos_eq[R]={math.degrees(qpos_eq[qadr_R]):.2f}deg "
              f"qvel_eq_norm={np.linalg.norm(qvel_eq):.4f} actual_qpos[R]={math.degrees(data.qpos[qadr_R]):.2f}deg")
        n = int(seconds / 0.01)
        for tick in range(n):
            if relin_during and tick > 0 and tick % int(RELIN_INTERVAL / 0.01) == 0:
                # 램프/정착 중에도 주기적으로 재선형화 - 하나의 K를
                # 너무 오래(수 초 이상) 고정해서 쓰면 그 자체로 서서히
                # 발산하는 경우가 있었다(오늘 찾은 버그) - override
                # 대상인 두 관절도 같이 갱신되지만, 바로 다음 줄에서
                # 다시 override로 덮어쓰니 문제 없음.
                qpos_eq = data.qpos.copy()
                qvel_eq = np.zeros_like(qvel_eq0)
                ctrl_trim = data.ctrl.copy()
                K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
            bias = bias_fn(tick, n)
            hip_L_cmd = min(bias, 0.0)
            hip_R_cmd = max(bias, 0.0)
            # 어제 검증된 원본 레시피 (prev_bias만 "실제 달성값"으로
            # 고친 버전) - 기준점(qpos_eq)은 램프 중 그대로 두고,
            # ctrl 쪽만 해당 관절 절대값으로 덮어쓴다(K@x는 그대로
            # 섞임 - 완전 배제/기준점 추적 둘 다 오히려 더 나빴다).
            ctrl_trim_eff = ctrl_trim.copy()
            ctrl_trim_eff[a_abduct_L] = hip_L_cmd
            ctrl_trim_eff[a_abduct_R] = hip_R_cmd
            kx_R_dbg = 0.0
            for _ in range(2):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
                dv = data.qvel - qvel_eq
                x = np.concatenate([dq, dv])
                Kx = K @ x
                kx_R_dbg = Kx[a_abduct_L]
                u = np.clip(ctrl_trim_eff - Kx, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)
            t = common.tilt_deg(data.qpos)
            if tick % 20 == 0:
                zx, zy = common.contact_zmp(model, data)
                print(f"    [{verbose_tag}] t={data.time:.2f} bias={math.degrees(bias):.1f} "
                      f"zmp_y={zy:.3f} tilt={t:.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f} "
                      f"hip_L_override={math.degrees(hip_L_cmd):.1f} Kx_R={math.degrees(kx_R_dbg):.1f} "
                      f"actual_hip_L={math.degrees(data.qpos[qadr_R]):.1f}")
            if t > 55:
                fell = data.time
                print(f"  [{verbose_tag}] FELL at t={data.time:.2f}")
                return False
        return True

    for cycle in range(len(targets)):
        target_bias = math.radians(targets[cycle % len(targets)])

        # 1) 램프: bias가 prev_bias -> target_bias로 선형 변화.
        n_ramp = int(RAMP_T / 0.01)
        ok = run_segment(RAMP_T, lambda tick, n: prev_bias + (target_bias - prev_bias) * (tick / n), "ramp")
        if not ok:
            break

        # 2) 정착: bias를 target_bias에 고정한 채 같은 K로 5초 더 유지
        #    (아직 재선형화 안 함 - 속도가 진짜로 가라앉을 시간을 줌).
        ok = run_segment(PRE_SETTLE_T, lambda tick, n: target_bias, "presettle")
        if not ok:
            break

        # 3) 재선형화 시작 - 이제부터 1초마다 "지금 상태"를 새 기준으로.
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
            t = common.tilt_deg(data.qpos)
            if t > 55:
                fell = data.time
                break
        zx, zy = common.contact_zmp(model, data)
        print(f"cycle {cycle} (target={math.degrees(target_bias):.0f}deg) 끝: t={data.time:.2f} "
              f"zmp_y={zy:.3f} tilt={common.tilt_deg(data.qpos):.2f}")
        if fell:
            break
        # 다음 세그먼트의 시작점은 "의도했던 목표"가 아니라 hold가
        # 끝났을 때 "실제로 trim에 들어간 값"이어야 한다 - hold 중
        # 재선형화가 반복되면서 trim이 서서히 줄어들 수 있어서(README의
        # "체중이동이 지워지는 현상"), 그 차이를 무시하면 다음 램프
        # 시작에 큰 불연속이 생긴다(오늘 찾은 버그).
        prev_bias = ctrl_trim[a_abduct_L] if ctrl_trim[a_abduct_L] < 0 else ctrl_trim[a_abduct_R]

    if fell:
        print("FELL at t=%.2f" % fell)
    else:
        print("2바퀴(8구간) 전부 통과")


if __name__ == "__main__":
    main()
