"""view_tvlqr_step.py
----------------------
tvlqr_step.py의 결과(standing -> one-leg 평형점 전이가 TV-LQR로
2초 동안 안정적으로 통과하고, 그 뒤 한 발 지지로 ~2초 더 버티다가
4.28초에 쓰러지는 것)를 MuJoCo 뷰어로 직접 본다.

실행:
  python view_tvlqr_step.py

창을 닫으면 종료.
"""
import math
import time

import numpy as np
import mujoco
import mujoco.viewer

import common
from tvlqr_step import build_waypoints, linearize_at, slerp_lerp_quat


def main():
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    qpos_eq1 = np.load("best_one_leg_qpos_v4.npy")

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

    print("웨이포인트 선형화 중...")
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
    print("TV-LQR 게인 계산 완료 - 뷰어 시작")

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
            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_print >= 0.2:
                last_print = data.time
                tilt = common.tilt_deg(data.qpos)
                phase = "transition" if t < T_transition else "one-leg hold"
                print(f"  t={data.time:5.2f} [{phase}] tilt={tilt:6.2f} "
                      f"|v|={np.linalg.norm(data.qvel[:3]):.3f}")

            time.sleep(max(0.0, model.opt.timestep - 0.0005))


if __name__ == "__main__":
    main()
