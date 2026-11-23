"""settle_wide_stance.py
-------------------------
분석적(역기구학+가우스-뉴턴) 접근이 계속 "오른발이 땅에 안 닿아도
가속도=0"인 해로 수렴해버렸다(접촉을 강제하는 항이 없어서). 대신
이미 검증된 방법(view_weight_shift.py, continuous_relinearize3.py)
- K0 피드백을 켠 채로 목표를 천천히 램프시켜서 물리 엔진이 직접
접촉/균형을 풀게 둔다. 오른다리를 바깥으로(hip_abduct_R) 벌리는
동시에, 왼다리 쪽에 작은 체중이동 feedforward(검증된 안전 범위
~3~5도)를 같이 램프한다.
"""
import math

import numpy as np
import mujoco

import common
from pelvis_width_test import widen_pelvis
from foot_offset_test import set_feet
from kp_kv_test import LEG_ACTUATORS, scale_kp_kv


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


def build_best_model():
    model = common.load_model()
    widen_pelvis(model, 0.40)
    set_feet(model, 0.28, 0.16, 0.0)
    scale_kp_kv(model, LEG_ACTUATORS, 3.0)
    model.opt.timestep = 0.0005
    return model


def run_one(model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub,
            a_abduct_R, a_ankle_roll_R, a_abduct_L, foot_L_id, foot_R_id,
            theta_r_deg, weight_shift_l_deg, ramp_t, settle_t, verbose=False, save=False):
    nv = model.nv
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)
    fell = None

    for tick in range(int((ramp_t + settle_t) / 0.01)):
        s = min(1.0, data.time / ramp_t)
        ff_R = math.radians(theta_r_deg) * s
        ff_ankle = -ff_R
        ff_L = math.radians(weight_shift_l_deg) * s
        for _ in range(n_sub):
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
            dv = data.qvel - qvel_eq0
            x = np.concatenate([dq, dv])
            u = -K0 @ x
            u[a_abduct_R] += ff_R
            u[a_ankle_roll_R] += ff_ankle
            u[a_abduct_L] += ff_L
            u = np.clip(u, -ctrl_limits, ctrl_limits)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
        tilt = common.tilt_deg(data.qpos)
        if verbose and tick % 100 == 0:
            fz_L = foot_fz(model, data, foot_L_id)
            fz_R = foot_fz(model, data, foot_R_id)
            print(f"  t={data.time:.2f} s={s:.2f} tilt={tilt:.2f} fz_L={fz_L:.1f} fz_R={fz_R:.1f}")
        if tilt > 55 and fell is None:
            fell = data.time
            break

    fz_L = foot_fz(model, data, foot_L_id)
    fz_R = foot_fz(model, data, foot_R_id)
    foot_y = data.xpos[foot_R_id][1]
    if save and fell is None:
        np.save("best_wide_stance_qpos.npy", data.qpos.copy())
        np.save("best_wide_stance_qvel.npy", data.qvel.copy())
    return fell, fz_L, fz_R, foot_y


def main():
    model = build_best_model()
    n_sub = round(0.01 / model.opt.timestep)
    nu = model.nu
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
    Q, R = common.build_cost_two_leg(model)
    R = R * 2.0
    K0 = common.solve_dare(A0, B0, Q, R)
    ctrl_limits = np.array([math.radians(30.0)] * nu)

    a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")
    a_ankle_roll_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_roll_R")
    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    foot_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
    foot_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")

    orig_y = -0.400

    print("=== theta_R=6도, 더 오래(60초) 버티는지 (부호 수정: +4도로 왼발에 체중) ===")
    fell, fz_L, fz_R, foot_y = run_one(
        model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub,
        a_abduct_R, a_ankle_roll_R, a_abduct_L, foot_L_id, foot_R_id,
        theta_r_deg=6.0, weight_shift_l_deg=4.0, ramp_t=5.0, settle_t=60.0, verbose=True)
    print(f"결과: {'생존' if fell is None else f'fell@{fell:.2f}'} "
          f"fz_L={fz_L:.1f} fz_R={fz_R:.1f} foot_y={foot_y:.3f}(+{foot_y-orig_y:.3f}m)")

    print()
    print("=== theta_R 스윕(6/10/15/20/25/30도), 20초 (부호 수정) ===")
    for theta_r in [6.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
        fell, fz_L, fz_R, foot_y = run_one(
            model, K0, qpos_eq0, qvel_eq0, ctrl_limits, n_sub,
            a_abduct_R, a_ankle_roll_R, a_abduct_L, foot_L_id, foot_R_id,
            theta_r_deg=theta_r, weight_shift_l_deg=4.0, ramp_t=5.0, settle_t=15.0,
            save=(theta_r == 6.0))
        print(f"theta_R={theta_r:5.1f}: {'생존' if fell is None else f'fell@{fell:.2f}'} "
              f"fz_L={fz_L:.1f} fz_R={fz_R:.1f} foot_y={foot_y:.3f}(+{foot_y-orig_y:.3f}m)")


if __name__ == "__main__":
    main()
