"""find_one_leg_equilibrium.py
-------------------------------
지금까지의 결론: K0(두 발 서있기 게인)는 체중을 한쪽 발로 충분히
옮기는 것 자체를 구조적으로 거부한다 (-20도를 명령해도 실제로는
~8도까지만 허용). 재선형화나 체중이동 트릭으로는 이 벽을 못 넘는다.

그래서 이번엔 K0를 전혀 쓰지 않고, **상수 ctrl(고정 트림)만으로
한쪽 발을 들고도 정적으로 버티는 자세**를 직접 찾는다 - 두 발
서있기의 `find_equilibrium()`(ctrl 고정하고 settle_sec만큼 그냥
시뮬레이션 돌려서 수렴하는지 보는 것)과 똑같은 방식을, 한 발
지지에 맞는 ctrl을 찾을 때까지 그리드 탐색으로 반복한다.

파라미터:
- lean_deg: hip_abduct_L을 이 각도만큼 틀어서 몸 전체(오른다리)를
  오른쪽으로 기울인다 (체중을 오른발 위로 이동).
- ankle_roll_L_deg: 그만큼 발목을 반대로 틀어서 오른발 바닥이
  평평하게 땅에 붙어있도록 상쇄한다(대략 -lean_deg).
- knee_R_deg: 왼쪽 무릎을 굽혀 발을 확실히 공중에 띄운다.
- hip_R_deg: 왼쪽 고관절 피치(앞뒤) - 무게중심 조절용.

각 조합에 대해 qpos_eq0에서 시작해 이 ctrl을 고정값으로 3초간
시뮬레이션하고, 왼발이 실제로 들렸는지(fz_L~0)와 tilt/속도가
안정적으로 유지되는지를 점수로 평가한다.
"""
import math

import numpy as np
import mujoco

import common


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


def try_pose(model, qpos_eq0, qvel_eq0, a_abduct_L, a_ankle_roll_L,
             a_hip_R, a_knee_R, a_ankle_pitch_R, foot_R_body,
             lean_deg, ankle_roll_deg, knee_R_deg, hip_R_deg,
             sim_s=3.0, verbose=False):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq0
    data.qvel[:] = qvel_eq0
    mujoco.mj_forward(model, data)

    ctrl = np.zeros(model.nu)
    ctrl[a_abduct_L] = math.radians(lean_deg)
    ctrl[a_ankle_roll_L] = math.radians(ankle_roll_deg)
    ctrl[a_knee_R] = math.radians(knee_R_deg)
    ctrl[a_hip_R] = math.radians(hip_R_deg)
    ctrl[a_ankle_pitch_R] = 0.0
    data.ctrl[:] = ctrl

    max_tilt = 0.0
    ramp_s = 0.5
    for tick in range(int(sim_s / model.opt.timestep)):
        s = min(1.0, data.time / ramp_s)
        cur = data.ctrl.copy()
        cur[a_abduct_L] = math.radians(lean_deg) * s
        cur[a_ankle_roll_L] = math.radians(ankle_roll_deg) * s
        cur[a_knee_R] = math.radians(knee_R_deg) * s
        cur[a_hip_R] = math.radians(hip_R_deg) * s
        data.ctrl[:] = cur
        mujoco.mj_step(model, data)
        t = common.tilt_deg(data.qpos)
        max_tilt = max(max_tilt, t)
        if t > 60:
            break
    fz_final = foot_fz(model, data, foot_R_body)
    vmag = np.linalg.norm(data.qvel[:3])
    tilt_final = common.tilt_deg(data.qpos)
    if verbose:
        print(f"    lean={lean_deg} ankle={ankle_roll_deg} knee_R={knee_R_deg} hip_R={hip_R_deg}: "
              f"fz_L_final={fz_final:.2f} tilt_final={tilt_final:.2f} max_tilt={max_tilt:.2f} |v|={vmag:.4f}")
    return fz_final, tilt_final, max_tilt, vmag, data.qpos.copy(), data.qvel.copy(), ctrl.copy()


def main():
    model = common.load_model()
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)

    a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
    a_ankle_roll_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_roll_L")
    a_hip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_R")
    a_knee_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_R")
    a_ankle_pitch_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ankle_pitch_R")
    foot_R_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")

    print("=== 한 발 지지 평형점 탐색 (상수 ctrl, 3초 시뮬레이션) ===")
    best = None
    for lean_deg in [10, 15, 20, 25, 30, 35]:
        for knee_R_deg in [25, 35, 45]:
            ankle_roll_deg = -lean_deg
            hip_R_deg = 0.0
            fz_final, tilt_final, max_tilt, vmag, qpos_f, qvel_f, ctrl_f = try_pose(
                model, qpos_eq0, qvel_eq0, a_abduct_L, a_ankle_roll_L,
                a_hip_R, a_knee_R, a_ankle_pitch_R, foot_R_body,
                lean_deg, ankle_roll_deg, knee_R_deg, hip_R_deg, sim_s=3.0)
            lifted = fz_final < 1.0
            score = max_tilt + (0 if lifted else 1000) + vmag * 10
            print(f"lean={lean_deg:3d} knee_R={knee_R_deg:3d}: fz_L_final={fz_final:7.2f} "
                  f"lifted={lifted} tilt_final={tilt_final:6.2f} max_tilt={max_tilt:6.2f} |v|={vmag:.4f}")
            if best is None or score < best[0]:
                best = (score, lean_deg, knee_R_deg, ankle_roll_deg, hip_R_deg,
                        fz_final, tilt_final, max_tilt, vmag)

    print()
    print("=== 최적 후보 ===")
    print(best)


if __name__ == "__main__":
    main()
