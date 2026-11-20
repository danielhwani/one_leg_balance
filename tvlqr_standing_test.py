"""tvlqr_standing_test.py
--------------------------
TV-LQR의 웨이포인트별 K 전환(0.1초마다 바뀌는 piecewise-constant
게인) 구조 자체가 문제인지, 아니면 궤적 내용(스윙 다리 움직임)이
문제인지를 가른다. 모든 웨이포인트를 qpos_eq0(가만히 서있기)로
고정해서 "아무 동작도 안 하는데 TV-LQR 틀만 그대로" 돌려본다.
K0 하나로 300초 버틴다는 건 이미 검증됐으니, 이 테스트가 실패하면
문제는 궤적이 아니라 TV-LQR 구조(웨이포인트 전환) 자체다.
"""
import math

import numpy as np
import mujoco

import common


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
    ctrl_eq0 = np.zeros(nu)  # 서있기는 ctrl=0이 아니라 실제 trim이 있어야 함

    data0 = mujoco.MjData(model)
    data0.qpos[:] = qpos_eq0
    data0.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data0)
    ctrl_eq0 = data0.ctrl.copy()  # settle 후의 실제 ctrl (대부분 0 근처일 것)

    Q, R = common.build_cost_two_leg(model)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    T_transition = 4.0
    n_wp = 40
    # 모든 웨이포인트를 qpos_eq0로 고정 - 아무 동작도 안 함
    wps = [qpos_eq0.copy() for _ in range(n_wp + 1)]
    ctrl_wps = [ctrl_eq0.copy() for _ in range(n_wp + 1)]

    print("웨이포인트별 선형화 중 (전부 같은 점)...")
    A_list, B_list = [], []
    for q, c in zip(wps, ctrl_wps):
        A, B = linearize_at(model, q, c, nv, nu)
        A_list.append(A)
        B_list.append(B)

    P = Q.copy()
    K_list = [None] * (n_wp + 1)
    A_end, B_end = A_list[-1], B_list[-1]
    for _ in range(200):
        BtP = B_end.T @ P
        K_end = np.linalg.solve(R + BtP @ B_end, BtP @ A_end)
        P = Q + A_end.T @ P @ A_end - A_end.T @ P @ B_end @ K_end
    K_list[n_wp] = K_end
    for k in range(n_wp - 1, -1, -1):
        A, B = A_list[k], B_list[k]
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        K_list[k] = K
    print("TV-LQR 게인 계산 완료 (이론상 전부 같은 K여야 함)")

    # 혹시 웨이포인트마다 K가 미묘하게 다른지 확인
    k_diffs = [np.max(np.abs(K_list[i] - K_list[0])) for i in range(n_wp + 1)]
    print(f"K 편차(웨이포인트 간 최대 차이): max={max(k_diffs):.6f}")

    K1 = K_list[n_wp]
    dt_wp = T_transition / n_wp

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None
    duration_s, hold_s = T_transition, 10.0
    for tick in range(int((duration_s + hold_s) / 0.01)):
        t = data.time
        if t < T_transition:
            k = min(n_wp - 1, int(t / dt_wp))
            qpos_ref = qpos_eq0
            ctrl_ref = ctrl_eq0
            K = K_list[k]
        else:
            qpos_ref = qpos_eq0
            ctrl_ref = ctrl_eq0
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
        if tick % 10 == 0:
            print(f"  t={data.time:.2f} tilt={tilt:6.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f}")
        if tilt > 55:
            fell = data.time
            print(f"쓰러짐: t={fell:.2f}")
            break

    if fell is None:
        print(f"{duration_s + hold_s}초 동안 버팀!")


if __name__ == "__main__":
    main()
