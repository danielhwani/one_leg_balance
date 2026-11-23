"""tvlqr_weightgate.py
-----------------------
사용자 진단: "지금 안 되는 게 weight shifting이다 - 좌우 체중이동이
안 되는 거지. CoM 궤적을 추종하고 지지발 하중 변화를 모니터링하면서
TV-LQR 피드백을 적용해야 할 거야."

지금까지(tvlqr_step.py)는 궤적 진행도 s를 **시계(t/T_transition)**로
결정했다 - 진단(tvlqr_diagnose.py)에서 waist_roll/스탠스 리언이
자기 시계 기준 목표보다 훨씬 앞서 실제 체중이동을 먼저 끝내버리는
걸 봤는데, 이건 "시계"가 아니라 "실제 물리적 진행"이 기준이어야
한다는 신호다.

이번엔 s를 시계 대신 **실제 측정한 지지발(로봇 왼쪽) 접촉력**으로
정한다: 서있을 때(절반 하중)부터 한 발 평형점(전체 하중)까지 몇 %
전달됐는지를 실시간으로 재서, 그 비율을 s로 써서 웨이포인트를
선택한다. 체중이 실제로 안 넘어가면 궤적이 그 자리에서 기다리고,
빨리 넘어가면 빨리 진행한다 - 시계가 아니라 물리가 페이스를 정함.
"""
import math

import numpy as np
import mujoco

import common
from tvlqr_step import build_waypoints, linearize_at, slerp_lerp_quat


def foot_fz(model, data, body_ids):
    """body_ids: 다리 하나의 하중을 재려면 foot뿐 아니라 shin도 같이
    넣어야 한다 - 이 모델은 정강이(shin) 캡슐도 땅에 닿아서 하중을
    받는다(발 72.56N + 정강이 50.6N = 123.16N = 전체 몸무게의 정확히
    절반, 발만 보면 절반에 한참 못 미치는 72.56N으로 잘못 읽힌다)."""
    if isinstance(body_ids, int):
        body_ids = [body_ids]
    total = 0.0
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if b1 not in body_ids and b2 not in body_ids:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        frame = c.frame.reshape(3, 3)
        total += (frame.T @ force[:3])[2]
    return total


def main():
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    qpos_eq1 = np.load("best_one_leg_qpos_v5.npy")

    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")  # 스탠스(로봇 왼쪽)
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")  # 스윙(로봇 오른쪽)
    shin_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shin_L")
    shin_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shin_R")
    stance_ids = [foot_L_id, shin_L_id]
    swing_ids = [foot_R_id, shin_R_id]

    data0 = mujoco.MjData(model)
    data0.qpos[:] = qpos_eq0
    data0.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data0)
    fz_stance_standing = foot_fz(model, data0, stance_ids)

    data1 = mujoco.MjData(model)
    data1.qpos[:] = qpos_eq1
    mujoco.mj_forward(model, data1)
    fz_stance_at_eq = foot_fz(model, data1, stance_ids)
    total_weight = sum(model.body_mass) * 9.81
    # v5 평형점(|qacc_free|=3.75, 덜 수렴됨)은 실제로 스탠스 다리가
    # 전체 몸무게를 다 받지 못한다(139N/246N) - 잔차 가속도가 남아
    # 있다는 뜻. 게이팅 목표는 "진짜 완전히 체중이 넘어간 상태"인
    # 전체 몸무게를 기준으로 삼는다(평형점 자체의 미수렴 오차에
    # 게이팅 민감도가 흔들리지 않도록).
    fz_stance_target = total_weight
    print(f"서있을 때 fz_stance={fz_stance_standing:.2f}, "
          f"한 발 평형점에서 실측 fz_stance={fz_stance_at_eq:.2f} (참고), "
          f"전체 몸무게={total_weight:.2f} (게이팅 목표로 사용)")

    def ctrl_of(qpos):
        c = np.zeros(nu)
        for i in range(nu):
            jid = model.actuator_trnid[i, 0]
            qadr = model.jnt_qposadr[jid]
            c[i] = qpos[qadr]
        return c

    Q, R = common.build_cost_two_leg(model)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    swing_joint_names = ["hip_abduct_R", "hip_R", "knee_R"]
    SWING_TRACK_WEIGHT = 30.0
    for jname in swing_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dofadr = model.jnt_dofadr[jid]
        Q[dofadr, dofadr] *= SWING_TRACK_WEIGHT
        Q[nv + dofadr, nv + dofadr] *= SWING_TRACK_WEIGHT

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
    print("TV-LQR 게인 계산 완료")

    # s가 1 틱에 너무 급하게 널뛰지 않도록(접촉력은 노이즈가 있다)
    # 저역통과 필터를 살짝 걸고, s는 뒤로는 못 가게(단조증가) 한다.
    S_SMOOTH = 0.15  # 1차 저역통과 시정수(초)

    def run(tag="", max_duration_s=10.0, verbose_every=10):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos_eq0
        data.qvel[:] = qvel_eq0
        mujoco.mj_forward(model, data)
        fell = None
        s_filt = 0.0
        s_max_reached = 0.0
        for tick in range(int(max_duration_s / 0.01)):
            fz_stance = foot_fz(model, data, stance_ids)
            s_meas = (fz_stance - fz_stance_standing) / (fz_stance_target - fz_stance_standing)
            s_meas = max(0.0, min(1.0, s_meas))
            alpha = 0.01 / (S_SMOOTH + 0.01)
            s_filt = s_filt + alpha * (s_meas - s_filt)
            s_max_reached = max(s_max_reached, s_filt)  # 단조증가 - 뒤로 안 감
            s_use = s_max_reached

            k = min(n_wp - 1, int(s_use * n_wp))
            s_local = s_use * n_wp - k
            qpos_ref = (1 - s_local) * wps[k] + s_local * wps[k + 1]
            qpos_ref[3:7] = slerp_lerp_quat(wps[k][3:7], wps[k + 1][3:7], s_local)
            ctrl_ref = (1 - s_local) * ctrl_wps[k] + s_local * ctrl_wps[k + 1]
            K = K_list[k] if s_use < 1.0 else K1

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
                print(f"  [{tag}] t={data.time:.2f} s_use={s_use:.3f} fz_stance={fz_stance:7.2f} "
                      f"tilt={tilt:6.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f}")
            if tilt > 55:
                fell = data.time
                print(f"[{tag}] 쓰러짐: t={fell:.2f} (s_use={s_use:.3f})")
                break
        if fell is None:
            print(f"[{tag}] {max_duration_s}초 동안 버팀! (최종 s_use={s_max_reached:.3f})")
        return fell

    print()
    print("=== 하중 게이팅 TV-LQR ===")
    run(tag="weightgate", max_duration_s=10.0, verbose_every=10)


if __name__ == "__main__":
    main()
