"""tvlqr_step.py
-----------------
오늘 결론: LQR은 "하나의 고정된 평형점 근처의 작은 편차"에서만
유효하다 - 체중이동/스텝처럼 큰 전이 자체는 안 됨 (K0도, 전용으로
새로 만든 K1도 실패). 사용자 제안: 고정 평형점 하나 대신, 서있기
(qpos_eq0) -> 한 발 지지 평형점(qpos_eq1, 오늘 가우스-뉴턴으로
찾음, |qacc_free|=0.18) 사이를 잇는 **궤적**을 만들고, 그 궤적을
따라가며 매 순간 그 지점에서 선형화한 시간 가변 게인 K(t)로
추적한다 (TV-LQR / finite-horizon LQR). "큰 동작"이지만 매 순간은
그 순간 목표점에서 작은 편차만 수정하면 되므로, 오늘 확인한
"LQR은 작은 편차에서만 유효하다"는 제약을 지키면서 큰 동작을
커버할 수 있는지 검증한다.

실패하면(2안) capture point/ZMP 기반 foot placement로 전환.
"""
import math

import numpy as np
import mujoco

import common


def slerp_lerp_quat(q0, q1, s):
    q = (1 - s) * q0 + s * q1
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else q0


def build_waypoints(qpos_eq0, qpos_eq1, n_wp):
    """standing -> one-leg 평형점 사이를 n_wp개 웨이포인트로 보간.
    smoothstep(3s^2-2s^3)로 가속/감속을 부드럽게."""
    wps = []
    for i in range(n_wp + 1):
        s_lin = i / n_wp
        s = 3 * s_lin**2 - 2 * s_lin**3
        qpos = (1 - s) * qpos_eq0 + s * qpos_eq1
        qpos[3:7] = slerp_lerp_quat(qpos_eq0[3:7], qpos_eq1[3:7], s)
        wps.append(qpos)
    return wps


def linearize_at(model, qpos, ctrl, nv, nu):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    return A, B


def main():
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    qpos_eq1 = np.load("best_one_leg_qpos_v4.npy")

    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")

    # 왼발 clearance 확인 (충돌을 켠 채로, qpos_eq1에서 왼발이 실제로
    # 땅 위에 떠 있는지)
    data_chk = mujoco.MjData(model)
    data_chk.qpos[:] = qpos_eq1
    mujoco.mj_forward(model, data_chk)
    print(f"qpos_eq1에서 foot_R z={data_chk.xpos[foot_R_id][2]:.4f}, "
          f"foot_L z={data_chk.xpos[foot_L_id][2]:.4f}")

    def ctrl_of(qpos):
        c = np.zeros(nu)
        for i in range(nu):
            jid = model.actuator_trnid[i, 0]
            qadr = model.jnt_qposadr[jid]
            c[i] = qpos[qadr]
        return c

    Q, R = common.build_cost_two_leg(model)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    T_transition = 4.0
    n_wp = 40
    wps = build_waypoints(qpos_eq0, qpos_eq1, n_wp)
    ctrl_wps = [ctrl_of(q) for q in wps]

    print("웨이포인트별 선형화 중...")
    A_list, B_list = [], []
    for q, c in zip(wps, ctrl_wps):
        A, B = linearize_at(model, q, c, nv, nu)
        A_list.append(A)
        B_list.append(B)
    print(f"{len(wps)}개 웨이포인트 선형화 완료")

    # 종단 비용: 마지막 웨이포인트(한 발 평형점)의 steady-state P
    # (K1 = 두 발 Q/R로 푼 DARE 그대로 재사용)
    P = Q.copy()
    K_list = [None] * (n_wp + 1)
    # steady-state로 몇 번 더 반복해서 종단 P를 K1 수준으로 맞춰준다
    A_end, B_end = A_list[-1], B_list[-1]
    for _ in range(200):
        BtP = B_end.T @ P
        K_end = np.linalg.solve(R + BtP @ B_end, BtP @ A_end)
        P = Q + A_end.T @ P @ A_end - A_end.T @ P @ B_end @ K_end
    K_list[n_wp] = K_end

    print("역방향 Riccati 재귀로 K(t) 계산 중...")
    for k in range(n_wp - 1, -1, -1):
        A, B = A_list[k], B_list[k]
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        K_list[k] = K
    print("TV-LQR 게인 계산 완료")

    # steady-state K1 (전이 끝난 뒤 계속 유지용)
    K1 = K_list[n_wp]

    dt_wp = T_transition / n_wp

    def run(duration_s, hold_s, verbose_every=10, tag=""):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos_eq0
        data.qvel[:] = qvel_eq0
        mujoco.mj_forward(model, data)
        fell = None
        for tick in range(int((duration_s + hold_s) / 0.01)):
            t = data.time
            if t < T_transition:
                k = min(n_wp - 1, int(t / dt_wp))
                s_local = (t - k * dt_wp) / dt_wp
                qpos_ref = (1 - s_local) * wps[k] + s_local * wps[k + 1]
                qpos_ref[3:7] = slerp_lerp_quat(wps[k][3:7], wps[k + 1][3:7], s_local)
                ctrl_ref = (1 - s_local) * ctrl_wps[k] + s_local * ctrl_wps[k + 1]
                K = K_list[k]
            else:
                qpos_ref = qpos_eq1
                ctrl_ref = ctrl_wps[-1]
                K = K1

            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_ref, data.qpos)
            dv = data.qvel - 0.0
            x = np.concatenate([dq, dv])
            u = ctrl_ref - K @ x
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            for _ in range(2):
                mujoco.mj_step(model, data)

            tilt = common.tilt_deg(data.qpos)
            if tick % verbose_every == 0:
                print(f"  [{tag}] t={data.time:.2f} tilt={tilt:6.2f} "
                      f"|v|={np.linalg.norm(data.qvel[:3]):.4f}")
            if tilt > 55:
                fell = data.time
                print(f"[{tag}] 쓰러짐: t={fell:.2f}")
                break
        if fell is None:
            print(f"[{tag}] {duration_s + hold_s}초 동안 버팀!")
        return fell

    print()
    print("=== TV-LQR로 standing -> one-leg 전이 (왼발 충돌 켠 채로) ===")
    run(T_transition, 5.0, verbose_every=10, tag="tvlqr")


if __name__ == "__main__":
    main()
