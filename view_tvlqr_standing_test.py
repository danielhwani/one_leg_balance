"""view_tvlqr_standing_test.py
-------------------------------
tvlqr_standing_test.py(TV-LQR 웨이포인트 전환 틀만 쓰고 모든
웨이포인트를 가만히 서있기로 고정 - 헤드리스로는 14초 완전 안정
확인됨)을 뷰어로 직접 본다.

실행:
  python view_tvlqr_standing_test.py

창을 닫으면 종료.
"""
import math
import time

import numpy as np
import mujoco
import mujoco.viewer

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

    data0 = mujoco.MjData(model)
    data0.qpos[:] = qpos_eq0
    data0.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data0)
    ctrl_eq0 = data0.ctrl.copy()

    Q, R = common.build_cost_two_leg(model)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    T_transition = 4.0
    n_wp = 40
    wps = [qpos_eq0.copy() for _ in range(n_wp + 1)]
    ctrl_wps = [ctrl_eq0.copy() for _ in range(n_wp + 1)]

    print("웨이포인트별 선형화 중 (전부 같은 점 - 가만히 서있기)...")
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
    K1 = K_list[n_wp]
    dt_wp = T_transition / n_wp
    print("TV-LQR 게인 계산 완료 - 뷰어 시작 (K가 0.1초마다 바뀌지만 전부 같은 목표)")

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    last_print = -1.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 7.5
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        while viewer.is_running():
            t = data.time
            if t < T_transition:
                k = min(n_wp - 1, int(t / dt_wp))
                K = K_list[k]
            else:
                K = K1
            qpos_ref = qpos_eq0
            ctrl_ref = ctrl_eq0

            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_ref, data.qpos)
            dv = data.qvel - 0.0
            x = np.concatenate([dq, dv])
            u = ctrl_ref - K @ x
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_print >= 0.2:
                last_print = data.time
                tilt = common.tilt_deg(data.qpos)
                print(f"  t={data.time:5.2f} tilt={tilt:6.2f} "
                      f"|v|={np.linalg.norm(data.qvel[:3]):.4f}")

            time.sleep(max(0.0, model.opt.timestep - 0.0005))


if __name__ == "__main__":
    main()
