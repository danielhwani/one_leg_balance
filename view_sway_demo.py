"""view_sway_demo.py
---------------------
오늘(2026-11-19) 최종적으로 찾은 가장 깨끗한 레시피를 MuJoCo 뷰어로
보여주는 데모. ROS2 없이 순수 터미널 + MuJoCo 창만 쓴다.

핵심 발견: K0(원래 두 발 서있기 LQR)는 hip_abduct_R/L 오프셋
±3~4도, 전진 push 0.25m/s까지는 "그냥 그대로" 잘 버틴다
(`compare_offset_vs_push.py`, `check_left_basin.py`로 검증). 그러니
기울기 목표를 20도가 아니라 **10도**로 잡으면, 기울고 유지한 뒤
돌아왔을 때의 잔여 오프셋(~3도)이 K0의 원래 유효 범위 안에 들어와서,
"영원히 재선형화"나 "짧게 밀기" 같은 복잡한 트릭 없이 **그냥 K0로
전환하는 것만으로 똑바로 서진다** (오히려 시간이 지날수록 기울기/
오프셋이 더 0에 가까워짐 - `try_k0_at_center.py`에서 120초 검증).

구조(오른쪽 -> 왼쪽 순서로 똑같은 레시피 반복):
1. 한쪽으로 10도 기울기(램프 5초 -> 정착 5초, K0 기반 override)
   -> 그 자세를 재선형화하며 10초 유지.
2. 유지가 끝나면(이 시점 오프셋이 이미 K0 범위 안) 바로 K0로
   전환하고 몇 초간 똑바로 서있는 걸 보여준 뒤, 반대쪽으로 반복.
3. 마지막엔 계속 K0만 사용(재선형화도 override도 없음).
"""
import math
import time

import numpy as np
import mujoco
import mujoco.viewer

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

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    qadr_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_R")]
    qadr_L = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]

    LEAN_DEG = 10.0  # K0 자체 범위(±3~4도) 안에 들어오도록 작게
    RAMP_T = 5.0
    PRE_SETTLE_T = 5.0
    HOLD_T = 10.0
    # 양쪽 기울기 사이 K0로 똑바로 서있는 구간 - 반대쪽 다리(hip_L
    # 등)에 남은 잔여 오프셋이 다음 기울기 전에 충분히 가라앉을
    # 시간이 필요하다(8초/30초는 부족해서 두 번째 기울기가 쓰러짐 -
    # 오늘 찾은 버그, README 참고). 60초는 검증됨.
    STAND_T = 60.0
    RELIN_INTERVAL = 1.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    print("오른쪽 -> 왼쪽으로 번갈아 살짝(10도) 기울기 -> 유지 -> K0로 "
          "똑바로 서기 데모. 창을 닫으면 종료.")

    tick_counter = [0]
    PRINT_EVERY_SEC = 1.0
    PRINT_INTERVAL_TICKS = max(1, int(PRINT_EVERY_SEC / 0.01))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        def step(K_, qpos_eq_, qvel_eq_, ctrl_trim_, act_id=None, cmd=None, realtime=True):
            ctrl_trim_eff = ctrl_trim_.copy()
            if act_id is not None:
                ctrl_trim_eff[act_id] = cmd
            for _ in range(2):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq_, data.qpos)
                dv = data.qvel - qvel_eq_
                x = np.concatenate([dq, dv])
                u = np.clip(ctrl_trim_eff - K_ @ x, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)
            viewer.sync()
            if realtime:
                time.sleep(max(0.0, model.opt.timestep - 0.0005))

            tick_counter[0] += 1
            if tick_counter[0] % PRINT_INTERVAL_TICKS == 0:
                print(f"    t={data.time:6.2f}s  기울기={common.tilt_deg(data.qpos):5.2f}도  "
                      f"엉덩이R={math.degrees(data.qpos[qadr_R]):+6.2f}도  "
                      f"엉덩이L={math.degrees(data.qpos[qadr_L]):+6.2f}도")

        def lean_and_return(act_id, sign, label):
            """sign=-1 -> 오른쪽(hip_abduct_R, 음수가 오른쪽 체중이동),
            sign=+1 -> 왼쪽(hip_abduct_L, 양수가 왼쪽 체중이동)."""
            K, qpos_eq, qvel_eq, ctrl_trim = K0, qpos_eq0, qvel_eq0, np.zeros(nu)

            print(f"{label}으로 기울기...")
            n_ramp = int(RAMP_T / 0.01)
            for tick in range(n_ramp):
                if not viewer.is_running():
                    return False
                bias = sign * math.radians(LEAN_DEG) * (tick / n_ramp)
                step(K, qpos_eq, qvel_eq, ctrl_trim, act_id, bias)

            n_settle = int(PRE_SETTLE_T / 0.01)
            for tick in range(n_settle):
                if not viewer.is_running():
                    return False
                step(K, qpos_eq, qvel_eq, ctrl_trim, act_id, sign * math.radians(LEAN_DEG))

            # 유지 (재선형화)
            qpos_eq = data.qpos.copy()
            qvel_eq = np.zeros_like(qvel_eq0)
            ctrl_trim = data.ctrl.copy()
            K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
            n_hold = int(HOLD_T / 0.01)
            for tick in range(n_hold):
                if not viewer.is_running():
                    return False
                if tick % int(RELIN_INTERVAL / 0.01) == 0:
                    qpos_eq = data.qpos.copy()
                    ctrl_trim = data.ctrl.copy()
                    K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
                step(K, qpos_eq, qvel_eq, ctrl_trim)

            print(f"{label} 유지 끝 - K0로 전환, 똑바로 서는 중... "
                  f"(별 움직임이 없는 구간이라 빠르게 넘어갑니다)")
            n_stand = int(STAND_T / 0.01)
            for tick in range(n_stand):
                if not viewer.is_running():
                    return False
                # 서있기만 하는 구간은 딱히 볼 게 없으니 실시간 재생
                # 없이 빠르게 넘긴다(그래도 매 틱 viewer.sync()는 하니
                # 창에는 계속 그려짐) - 60초가 실제로 60초 걸리면
                # "멈춘 것처럼" 보인다는 사용자 피드백으로 고침.
                step(K0, qpos_eq0, qvel_eq0, np.zeros(nu), realtime=False)
            return True

        if not lean_and_return(a_abduct_R, -1.0, "오른쪽"):
            return
        if not lean_and_return(a_abduct_L, 1.0, "왼쪽"):
            return

        print("데모 끝 - 계속 똑바로 서있는 모습을 보실 수 있습니다 (창을 닫으면 종료).")
        while viewer.is_running():
            step(K0, qpos_eq0, qvel_eq0, np.zeros(nu))


if __name__ == "__main__":
    main()
