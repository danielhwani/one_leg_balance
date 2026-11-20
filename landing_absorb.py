"""landing_absorb.py
---------------------
open_loop_swing.py에서 스윙 자체는 부드러워졌지만(전신 피드백을 꺼서
"이동"으로만 취급), 착지 순간 다리가 거의 펴진 채(목표 무릎각 0도)
뻣뻣하게 닿아 평소 6배(436N) 충격력이 튀고, 그 충격에 발이 다시
튕겨 접촉을 잃으면서 쓰러졌다.

사용자 제안: 착지 시 발목/무릎을 굽혀서 충격을 흡수한 다음 균형을
잡도록 착지한 발의 관절을 정교하게 조정하고, 필요하면 몸통/팔도
보조 역할을 하게 한다.

구조 (3단계):
1) stand: 순수 K0.
2) swing: 전신 피드백 끔(이륙 시점 trim 유지) + 스윙 다리(hip_L/knee_L)만
   스크립트. 착지 목표 무릎각을 0도가 아니라 **20도(이미 굽은 채)**로
   바꿔 애초에 뻣뻣하게 닿지 않게 한다.
3) 접촉력이 일정 틱 이상 유지되면(dwell, 순간 바운스 오탐 방지) absorb로
   전환: 착지 다리(knee_L, ankle_pitch_L)는 명시적 흡수 프로파일
   (더 깊이 굽혔다가 다시 펴서 안정 자세로) 스크립트, **나머지 전신은
   K0(원래 평형점 기준) 피드백 재개** - 몸통/팔이 그 피드백으로 자연히
   보조 역할을 하게 둔다.
4) absorb 종료 + 속도가 충분히 가라앉으면 landed로 전환, 그 시점 상태로
   재선형화한 전용 K로 넘어간다.
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


def foot_fz(model, data, body_id):
    total = 0.0
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if b1 != body_id and b2 != body_id:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        frame = c.frame.reshape(3, 3)
        total += (frame.T @ force[:3])[2]
    return total


def run(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits,
        a_hip_L, a_knee_L, a_ankle_pitch_L, foot_L_body,
        lift_knee_deg, land_knee_deg, hip_target_deg,
        swing_duration_s,
        touch_fz=5.0, touch_dwell_s=0.02,
        absorb_deep_deg=35.0, absorb_down_s=0.12, absorb_up_s=0.25,
        settle_v_thresh=0.3,
        post_relin_interval=0.1,
        duration_s=15.0, verbose=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    phase = "stand"  # stand -> swing -> absorb -> landed
    stand_s = 0.2
    swing_t0 = None
    trim_at_liftoff = None
    touch_t0 = None       # fz 처음 threshold 넘은 시각(바운스일 수도)
    touch_confirmed_t = None
    has_lifted = False    # 실제 이륙(fz~0)을 거친 적이 있는지 - 이게 없으면
                           # "아직 안 뗀 발"을 착지로 오인한다
    absorb_t0 = None
    landed_t = None
    fell = None
    K, qpos_eq, qvel_eq, ctrl_trim = K0, qpos_eq0, qvel_eq0, np.zeros(model.nu)
    tick_since_relin = 0
    knee_at_touch = 0.0

    for tick in range(int(duration_s / 0.01)):
        fz = foot_fz(model, data, foot_L_body)
        vmag = np.linalg.norm(data.qvel[:3])

        if phase == "stand" and data.time >= stand_s:
            phase = "swing"
            swing_t0 = data.time
            trim_at_liftoff = data.ctrl.copy()

        elif phase == "swing":
            if not has_lifted:
                if fz < 1.0:
                    has_lifted = True
            elif fz > touch_fz:
                if touch_t0 is None:
                    touch_t0 = data.time
                elif data.time - touch_t0 >= touch_dwell_s:
                    phase = "absorb"
                    touch_confirmed_t = touch_t0
                    absorb_t0 = data.time
                    knee_at_touch = data.ctrl[a_knee_L]
            else:
                touch_t0 = None  # 재부상 - dwell 리셋

        elif phase == "absorb":
            absorb_total = absorb_down_s + absorb_up_s
            if (data.time - absorb_t0) >= absorb_total and vmag < settle_v_thresh:
                phase = "landed"
                landed_t = data.time
                tick_since_relin = 0

        if phase == "landed":
            if tick_since_relin % int(post_relin_interval / 0.01) == 0:
                qpos_eq = data.qpos.copy()
                qvel_eq = np.zeros_like(qvel_eq0)
                ctrl_trim = data.ctrl.copy()
                K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
            tick_since_relin += 1

        for _ in range(2):
            if phase == "stand":
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
            elif phase == "swing":
                u = trim_at_liftoff.copy()
            elif phase == "absorb":
                # 전신은 K0(원래 평형점 기준) 재개 - 몸통/팔이 보조
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
                dv = data.qvel - qvel_eq0
                x = np.concatenate([dq, dv])
                u = -K0 @ x
            else:  # landed
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
                dv = data.qvel - qvel_eq
                x = np.concatenate([dq, dv])
                u = ctrl_trim - K @ x

            if phase == "swing":
                s = min(1.0, (data.time - swing_t0) / swing_duration_s)
                u[a_hip_L] = math.radians(hip_target_deg) * s
                u[a_knee_L] = math.radians(lift_knee_deg) * (1 - abs(2 * s - 1)) \
                    + math.radians(land_knee_deg) * max(0.0, 2 * s - 1)
                u[a_ankle_pitch_L] = 0.0
            elif phase == "absorb":
                # 착지 다리만 명시적 흡수 프로파일로 덮어씀(무릎: 더 굽혔다가 펴기)
                dt = data.time - absorb_t0
                if dt < absorb_down_s:
                    sd = dt / absorb_down_s
                    knee_at_touch_deg = math.degrees(knee_at_touch)
                    target = knee_at_touch_deg + (absorb_deep_deg - knee_at_touch_deg) * sd
                else:
                    su = min(1.0, (dt - absorb_down_s) / absorb_up_s)
                    target = absorb_deep_deg + (land_knee_deg - absorb_deep_deg) * su
                u[a_knee_L] = math.radians(target)
                u[a_ankle_pitch_L] = 0.0

            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        t = common.tilt_deg(data.qpos)
        if verbose and tick % 10 == 0:
            print(f"  t={data.time:.2f} phase={phase:7s} fz_L={fz:7.2f} "
                  f"tilt={t:5.2f} |v|={vmag:.4f}")
        if t > 55 and fell is None:
            fell = data.time
            break

    return touch_confirmed_t, landed_t, fell


def main():
    model = common.load_model()
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_hip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_L")
    a_knee_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_L")
    a_ankle_pitch_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_pitch_L")
    foot_L_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")

    print("=== 착지 흡수(무릎 굽혀 쿠션) 추가 - swing_duration x land_knee 스윕 ===")
    for swing_dur in [0.5, 0.8, 1.2]:
        for land_knee in [15.0, 20.0, 25.0]:
            touch, landed, fell = run(
                model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits,
                a_hip_L, a_knee_L, a_ankle_pitch_L, foot_L_body,
                lift_knee_deg=25.0, land_knee_deg=land_knee, hip_target_deg=15.0,
                swing_duration_s=swing_dur, duration_s=15.0)
            print(f"swing_dur={swing_dur:4.1f} land_knee={land_knee:5.1f}: "
                  f"touch={touch} landed={landed} fell={fell}")

    print()
    print("=== 상세 추적: swing_dur=0.8, land_knee=20 ===")
    run(model, K0, qpos_eq0, qvel_eq0, Q, R, ctrl_limits,
        a_hip_L, a_knee_L, a_ankle_pitch_L, foot_L_body,
        lift_knee_deg=25.0, land_knee_deg=20.0, hip_target_deg=15.0,
        swing_duration_s=0.8, duration_s=10.0, verbose=True)


if __name__ == "__main__":
    main()
