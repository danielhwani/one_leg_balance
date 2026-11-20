"""terminal_demo.py
--------------------
MuJoCo GUI 창 없이, 일반 터미널에서 텍스트로 보는 데모. 오늘
(2026-11-19) 찾은 레시피 - "10도로 살짝 기울기 -> 유지(재선형화)
-> K0로 똑바로 서기"를 오른쪽, 왼쪽 양쪽으로 보여준다.

README의 "핵심 요약" 섹션 참고:
- 기울기는 외력이 아니라 관절(hip_abduct) 자체에 주는 명령.
- LQR 저항 때문에 명령(10도)의 일부만(~3도) 실제로 달성됨 - 그동안
  몸통 자체 기울기는 거의 0으로 유지됨.
- 그 "실제 도달한 지점"에서 1초마다 재선형화하며 유지.
- 유지가 끝나면 원래(맨 처음 똑바로 서있을 때 계산한) K0로 바로
  전환 - 그 지점이 K0의 실제 유효 범위(±3~4도) 안이라 저절로
  똑바른 자세 쪽으로 수렴한다.
- 양쪽을 연달아 하려면 반대쪽 다리에 남는 잔류 오프셋이 가라앉을
  시간(60초)이 필요하다(README 참고).

실행:
  python terminal_demo.py
"""
import math
import time

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


def main():
    print("모델 불러오는 중...")
    model = common.load_model()
    nv, nu = model.nv, model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    qadr_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]
    qadr_L = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_R")]

    LEAN_DEG, RAMP_T, PRE_SETTLE_T, HOLD_T, STAND_T, RELIN_INTERVAL = 10.0, 5.0, 5.0, 10.0, 60.0, 1.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    PRINT_EVERY = 1.0  # 몇 초마다 상태 한 줄 찍을지

    def step(K_, qpos_eq_, qvel_eq_, ctrl_trim_, act_id=None, cmd=None):
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

    def status_line():
        return (f"    t={data.time:6.2f}s  기울기={common.tilt_deg(data.qpos):5.2f}도  "
                f"엉덩이R={math.degrees(data.qpos[qadr_R]):+6.2f}도  "
                f"엉덩이L={math.degrees(data.qpos[qadr_L]):+6.2f}도")

    def run_phase(seconds, step_fn, print_every=PRINT_EVERY):
        n = int(seconds / 0.01)
        print_interval = max(1, int(print_every / 0.01))
        for tick in range(n):
            step_fn(tick, n)
            if tick % print_interval == 0:
                print(status_line())

    def lean_and_return(act_id, sign, label):
        K, qpos_eq, qvel_eq, ctrl_trim = K0, qpos_eq0, qvel_eq0, np.zeros(nu)

        print(f"\n=== {label}으로 살짝({LEAN_DEG:.0f}도) 기울기 ===")
        n_ramp = int(RAMP_T / 0.01)
        run_phase(RAMP_T, lambda tick, n: step(
            K, qpos_eq, qvel_eq, ctrl_trim, act_id, sign * math.radians(LEAN_DEG) * (tick / n_ramp)))
        run_phase(PRE_SETTLE_T, lambda tick, n: step(
            K, qpos_eq, qvel_eq, ctrl_trim, act_id, sign * math.radians(LEAN_DEG)))

        print(f"\n=== {label} 기울기 유지 (1초마다 재선형화) ===")
        nonlocal_K = [relinearize_at(model, data.qpos.copy(), data.ctrl.copy(), Q, R)]
        qpos_eq_box, ctrl_trim_box = [data.qpos.copy()], [data.ctrl.copy()]
        qvel_eq_hold = np.zeros_like(qvel_eq0)

        def hold_step(tick, n):
            if tick % int(RELIN_INTERVAL / 0.01) == 0:
                qpos_eq_box[0] = data.qpos.copy()
                ctrl_trim_box[0] = data.ctrl.copy()
                nonlocal_K[0] = relinearize_at(model, qpos_eq_box[0], ctrl_trim_box[0], Q, R)
            step(nonlocal_K[0], qpos_eq_box[0], qvel_eq_hold, ctrl_trim_box[0])

        run_phase(HOLD_T, hold_step)

        print(f"\n=== {label} 유지 끝 -> 원래 K0로 전환, 똑바로 서는 중 ({STAND_T:.0f}초) ===")
        run_phase(STAND_T, lambda tick, n: step(K0, qpos_eq0, qvel_eq0, np.zeros(nu)), print_every=5.0)

    lean_and_return(a_abduct_R, -1.0, "오른쪽")
    lean_and_return(a_abduct_L, 1.0, "왼쪽")

    print("\n데모 끝 - 계속 똑바로 서있는 상태입니다.")
    print(status_line())


if __name__ == "__main__":
    main()
