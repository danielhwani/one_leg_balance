"""staged_shift.py
------------------
continuous_relinearize3.py에서 성공한 메커니즘(재선형화마다 그 순간의
전체 ctrl을 trim으로 유지)을 그대로 쓰되, 목표를 한 번에 크게 주지
않고 **관절 물리 한계(hip_abduct ±25deg) 안에서 단계적으로** 올려서
얼마나 더 체중을 옮길 수 있는지 본다.
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
    Q, R = common.build_cost_two_leg(model)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    qadr_R = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_abduct_L")]

    STAGE_TARGETS_DEG = [-15, -18, -20, -22, -23, -24]
    RAMP_T = 3.0
    PRE_SETTLE_T = 5.0
    HOLD_T = 8.0
    RELIN_INTERVAL = 1.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0

    qpos_eq = qpos_eq0.copy()
    ctrl_trim = np.zeros(nu)
    K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
    prev_ff = 0.0
    global_tick = 0
    fell = None

    for stage_deg in STAGE_TARGETS_DEG:
        target = math.radians(stage_deg)
        stage_t0 = data.time
        for tick in range(int((RAMP_T + PRE_SETTLE_T + HOLD_T) / 0.01)):
            s = min(1.0, (data.time - stage_t0) / RAMP_T)
            ff = prev_ff + (target - prev_ff) * s
            past_settle = (data.time - stage_t0) > (RAMP_T + PRE_SETTLE_T)
            if past_settle and global_tick % int(RELIN_INTERVAL / 0.01) == 0:
                qpos_eq = data.qpos.copy()
                ctrl_trim = data.ctrl.copy()
                K = relinearize_at(model, qpos_eq, ctrl_trim, Q, R)
            for _ in range(2):
                dq = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq, data.qpos)
                dv = data.qvel - np.zeros_like(qvel_eq0)
                x = np.concatenate([dq, dv])
                if past_settle:
                    # 정착 후엔 다른 관절과 똑같이 "통째로 캡처한 trim"만
                    # 쓴다 - 이 관절만 따로 ff로 덮어쓰면(이전 시도의
                    # 버그) 그 관절의 trim 의미가 다른 관절과 달라져서
                    # 재선형화 시점에 불일치가 생긴다.
                    u = ctrl_trim - K @ x
                else:
                    ctrl_trim_eff = ctrl_trim.copy()
                    ctrl_trim_eff[a_abduct_L] = ff
                    u = ctrl_trim_eff - K @ x
                u = np.clip(u, -ctrl_limits, ctrl_limits)
                data.ctrl[:] = u
                mujoco.mj_step(model, data)
            global_tick += 1
            t = common.tilt_deg(data.qpos)
            if tick % 50 == 0:
                zx, zy = common.contact_zmp(model, data)
                print(f"    [{stage_deg}deg] t={data.time:.2f} s={s:.2f} past_settle={past_settle} "
                      f"ff={math.degrees(ff):.1f} zmp_y={zy:.3f} tilt={t:.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f}")
            if t > 55 and fell is None:
                fell = data.time
                break
        if fell is not None:
            break
        prev_ff = target
        zx, zy = common.contact_zmp(model, data)
        actual = math.degrees(data.qpos[qadr_R])
        print(f"stage={stage_deg}deg 끝(t={data.time:.2f}): actual={actual:.1f}deg zmp_y={zy:.3f} "
              f"tilt={common.tilt_deg(data.qpos):.2f} |v|={np.linalg.norm(data.qvel[:3]):.4f}")

    if fell is not None:
        print("FELL at t=%.2f" % fell)
    else:
        print("모든 단계 통과")


if __name__ == "__main__":
    main()
