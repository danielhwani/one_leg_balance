"""test_one_leg_lqr.py
-----------------------
find_one_leg_equilibrium_v3.py가 가우스-뉴턴으로 찾은 한 발 지지
평형점(best_one_leg_qpos.npy, |qacc_free|=0.18까지 수렴)을 진짜
평형점으로 써서:
1) 그 지점에서 새로 선형화(mjd_transitionFD)하고
2) 두 발 서있기와 같은 Q/R로 전용 LQR K1을 푼 다음
3) qvel=0에서 시작해 K1 피드백만으로 얼마나 오래 버티는지 헤드리스로
   확인한다 (두 발 서있기의 K0 검증과 같은 방식).

왼발은 이 평형점을 찾을 때처럼 충돌을 비활성화해둔 상태로 먼저
테스트한다 - "이상화된 한 발 지지"가 LQR로 안정화되는지부터 보고,
되면 그 다음에 왼발 충돌을 다시 켜고 테스트한다.
"""
import math

import numpy as np
import mujoco

import common


def main():
    model = common.load_model()
    nv, nu = model.nv, model.nu

    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
    foot_R_geom = model.body_geomadr[foot_R_id]
    model.geom_contype[foot_R_geom] = 0
    model.geom_conaffinity[foot_R_geom] = 0

    qpos_raw = np.load("best_one_leg_qpos.npy")

    ctrl_eq1 = np.zeros(nu)
    for i in range(nu):
        jid = model.actuator_trnid[i, 0]
        qadr = model.jnt_qposadr[jid]
        ctrl_eq1[i] = qpos_raw[qadr]

    # 동적으로(free dynamics) settle시키면 한 발 지지는 원래 불안정해서
    # 짧은 settle 중에도 자세가 드리프트해버린다(두 발 서있기와 다름 -
    # 거긴 거의 안정적인 자세라 0.3초 settle이 안전했음). 대신 접촉이
    # "정확히 침투=0" 경계에 있어서 선형화가 불안정해지는 문제만
    # 수직으로 아주 살짝(z만) 눌러서 피한다 - 자세/자유관절은 그대로.
    best_z = None
    for dz in [0.0, -0.0003, -0.0007, -0.0012, -0.002, -0.004, -0.008]:
        qpos_try = qpos_raw.copy()
        qpos_try[2] += dz
        data_try = mujoco.MjData(model)
        data_try.qpos[:] = qpos_try
        data_try.qvel[:] = 0.0
        data_try.ctrl[:] = ctrl_eq1
        mujoco.mj_forward(model, data_try)
        acc = np.linalg.norm(data_try.qacc[:6])
        print(f"  dz={dz:+.4f}: |qacc[:6]|={acc:.4f}")
        if best_z is None or acc < best_z[0]:
            best_z = (acc, dz, qpos_try.copy())

    qpos_eq1 = best_z[2]
    qvel_eq1 = np.zeros(nv)
    print(f"선택: dz={best_z[1]} |qacc[:6]|={best_z[0]:.4f}")

    data_lin = mujoco.MjData(model)
    data_lin.qpos[:] = qpos_eq1
    data_lin.qvel[:] = 0.0
    data_lin.ctrl[:] = ctrl_eq1
    mujoco.mj_forward(model, data_lin)
    print(f"선형화 지점 확인: |qacc[:6]|={np.linalg.norm(data_lin.qacc[:6]):.4f}")

    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data_lin, 1e-6, True, A, B, None, None)

    Q, R = common.build_cost_two_leg(model)
    K1 = common.solve_dare(A, B, Q, R)
    print("K1 계산 완료")

    ctrl_limits = np.array([math.radians(30.0)] * nu)

    def test_K(K, label, duration_s=10.0, verbose_every=10):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos_eq1
        data.qvel[:] = qvel_eq1
        mujoco.mj_forward(model, data)
        fell = None
        for tick in range(int(duration_s / 0.01)):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq1, data.qpos)
            dv = data.qvel - qvel_eq1
            x = np.concatenate([dq, dv])
            u_raw = ctrl_eq1 - K @ x
            u = np.clip(u_raw, -ctrl_limits, ctrl_limits)
            n_sat = int(np.sum(np.abs(u_raw) >= ctrl_limits - 1e-6))
            data.ctrl[:] = u
            for _ in range(2):
                mujoco.mj_step(model, data)
            t = common.tilt_deg(data.qpos)
            if tick % verbose_every == 0:
                print(f"  [{label}] t={data.time:.2f} tilt={t:.2f} "
                      f"|v|={np.linalg.norm(data.qvel[:3]):.4f} n_sat={n_sat}")
            if t > 55:
                fell = data.time
                print(f"[{label}] 쓰러짐: t={fell:.2f}")
                break
        if fell is None:
            print(f"[{label}] {duration_s}초 동안 버팀!")
        return fell

    print()
    print("=== 기본 K1 (두 발 서있기 Q/R 그대로) ===")
    test_K(K1, "K1_base", duration_s=5.0, verbose_every=10)

    print()
    print("=== Q를 더 강하게(자세/각속도 가중 10배) 한 K2 ===")
    Q2 = Q.copy()
    Q2[0:6, 0:6] *= 10.0
    Q2[nv:nv + 6, nv:nv + 6] *= 10.0
    K2 = common.solve_dare(A, B, Q2, R)
    test_K(K2, "K2_Q10x", duration_s=5.0, verbose_every=10)

    print()
    print("=== Q를 100배 한 K3 ===")
    Q3 = Q.copy()
    Q3[0:6, 0:6] *= 100.0
    Q3[nv:nv + 6, nv:nv + 6] *= 100.0
    K3 = common.solve_dare(A, B, Q3, R)
    test_K(K3, "K3_Q100x", duration_s=5.0, verbose_every=10)

    print()
    print("=== 액추에이터 한계를 풀어서(±80도) K1로 재시도 - 포화가 진짜 원인인지 진단 ===")
    ctrl_limits = np.array([math.radians(80.0)] * nu)
    test_K(K1, "K1_limit80", duration_s=8.0, verbose_every=10)

    print()
    print("=== 한계 없음(거의 무제한, ±300도)으로 K1 재시도 ===")
    ctrl_limits = np.array([math.radians(300.0)] * nu)
    test_K(K1, "K1_nolimit", duration_s=8.0, verbose_every=10)


if __name__ == "__main__":
    main()
