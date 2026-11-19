"""zmp_nonlinear_stand.py
--------------------------
LQR을 완전히 안 쓰고, 순수 ZMP 피드백만으로 두 발 서있기부터 도전.

- 발목(ankle_roll/pitch, 4개)만 ZMP 오차에 비례(PD)해서 능동적으로
  움직인다 - 이게 실제 2족 로봇에서 쓰는 고전적 "ankle strategy".
- 나머지 22개 액추에이터는 qpos_eq0의 자기 자신 목표값을 그대로
  유지(= 가만히 서있는 자세 그대로, 추가 피드백 없음).
- 평형점(qpos_eq0)에서 시작하는 것부터 확인하고, 그 다음 초기
  자세를 살짝씩 바꿔가며(hip_abduct_R 오프셋, 전체 기울기 등) 어디까지
  버티는지 폭을 스윕한다.
"""
import math

import numpy as np
import mujoco

import common


def build_natural_targets(model, qpos_eq0):
    """ankle_roll/pitch 4개를 제외한 모든 position 액추에이터의
    자연스러운(qpos_eq0) 목표값 - 각 액추에이터가 구동하는 조인트의
    qpos_eq0 값을 그대로 가져온다."""
    nu = model.nu
    targets = np.zeros(nu)
    for i in range(nu):
        jid = model.actuator_trnid[i, 0]
        qadr = model.jnt_qposadr[jid]
        targets[i] = qpos_eq0[qadr]
    return targets


def run(model, qpos_eq0, qvel_eq0, natural_targets, ankle_ids, hip_ids=None,
        kp_x=0.0, kd_x=0.0, kp_y=0.0, kd_y=0.0,
        kp_hip_x=0.0, kd_hip_x=0.0, kp_hip_y=0.0, kd_hip_y=0.0,
        init_qpos=None, init_qvel=None,
        duration=30.0, verbose=False):
    nv, nu = model.nv, model.nu
    data = mujoco.MjData(model)
    data.qpos[:] = init_qpos if init_qpos is not None else qpos_eq0
    data.qvel[:] = init_qvel if init_qvel is not None else qvel_eq0
    mujoco.mj_forward(model, data)

    a_roll_L, a_roll_R, a_pitch_L, a_pitch_R = ankle_ids
    if hip_ids is not None:
        a_hip_abduct_L, a_hip_abduct_R, a_hip_pitch_L, a_hip_pitch_R = hip_ids
    zmp_x_prev, zmp_y_prev = common.contact_zmp(model, data)
    fell = None

    for tick in range(int(duration / 0.01)):
        for _ in range(2):
            zmp_x, zmp_y = common.contact_zmp(model, data)
            zmp_x_rate = (zmp_x - zmp_x_prev) / model.opt.timestep
            zmp_y_rate = (zmp_y - zmp_y_prev) / model.opt.timestep
            zmp_x_prev, zmp_y_prev = zmp_x, zmp_y

            # 목표 ZMP = 0 근처(두 발 중앙) - 실제 qpos_eq0의 정지
            # ZMP도 거의 0이었다(check_zmp.py에서 확인).
            err_x = zmp_x - 0.0
            err_y = zmp_y - 0.0
            ankle_pitch_fb = -kp_x * err_x - kd_x * zmp_x_rate
            ankle_roll_fb = -kp_y * err_y - kd_y * zmp_y_rate

            u = natural_targets.copy()
            u[a_pitch_L] += ankle_pitch_fb
            u[a_pitch_R] += ankle_pitch_fb
            u[a_roll_L] += ankle_roll_fb
            u[a_roll_R] += ankle_roll_fb

            if hip_ids is not None:
                hip_pitch_fb = -kp_hip_x * err_x - kd_hip_x * zmp_x_rate
                hip_abduct_fb = -kp_hip_y * err_y - kd_hip_y * zmp_y_rate
                u[a_hip_pitch_L] += hip_pitch_fb
                u[a_hip_pitch_R] += hip_pitch_fb
                u[a_hip_abduct_L] += hip_abduct_fb
                u[a_hip_abduct_R] += hip_abduct_fb

            data.ctrl[:] = u
            mujoco.mj_step(model, data)

        t = common.tilt_deg(data.qpos)
        if verbose and tick % 100 == 0:
            print(f"  t={data.time:.2f} zmp=({zmp_x:.3f},{zmp_y:.3f}) tilt={t:.2f} "
                  f"|v|={np.linalg.norm(data.qvel[:3]):.4f}")
        if t > 55 and fell is None:
            fell = data.time
            break

    return fell


def main():
    model = common.load_model()
    qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
    natural_targets = build_natural_targets(model, qpos_eq0)

    def act(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    ankle_ids = (act("act_ankle_roll_L"), act("act_ankle_roll_R"),
                 act("act_ankle_pitch_L"), act("act_ankle_pitch_R"))
    # hip_abduct(좌우)는 양쪽에 같은 값을 주면 "쐐기" 효과로 이상하게
    # 반응한다는 걸 이미 알고 있어서(오늘 isolate_weight_shift_only.py)
    # 일단 hip_pitch(앞뒤, 양쪽 같은 방향이 자연스러운 동작)만 추가.
    hip_ids = (act("act_hip_abduct_L"), act("act_hip_abduct_R"),
               act("act_hip_L"), act("act_hip_R"))

    print("=== 발목 + 엉덩이 피치(앞뒤)까지 같이, 게인 스윕 ===")
    for kp in [0.1, 0.3, 0.5, 1.0]:
        for kp_hip in [0.0, 0.3, 1.0, 3.0]:
            fell = run(model, qpos_eq0, qvel_eq0, natural_targets, ankle_ids, hip_ids,
                       kp_x=kp, kd_x=0.0, kp_y=kp, kd_y=0.0,
                       kp_hip_x=kp_hip, kd_hip_x=0.0, kp_hip_y=0.0, kd_hip_y=0.0,
                       duration=10.0)
            print(f"kp_ankle={kp} kp_hip_pitch={kp_hip}: fell={fell}")


if __name__ == "__main__":
    main()
