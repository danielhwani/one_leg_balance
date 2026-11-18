"""common.py
-----------
외발 균형(one_leg_balance) 프로젝트의 공용 유틸리티. computer_graphics
프로젝트의 humanoid_mujoco_walk.py에서 이미 검증된 두 발 서있기
LQR 코드를 그대로 가져왔고(평형점 찾기/선형화/Riccati 풀이는 바꿀 필요
없음 - 두 발 서있기는 이미 해결된 문제), 접촉력 기반 ZMP 계산은 이번
프로젝트에서 새로 검증한 것을 추가했다.
"""
import math
import os

import numpy as np
import mujoco

MODEL_PATH = os.path.join(os.path.dirname(__file__), "mujoco", "humanoid_balance.xml")

EQUILIBRIUM_SETTLE_SEC = 0.3
GRAVITY = 9.81


def load_model():
    return mujoco.MjModel.from_xml_path(MODEL_PATH)


def find_equilibrium(model, settle_sec=EQUILIBRIUM_SETTLE_SEC):
    data = mujoco.MjData(model)
    for _ in range(int(settle_sec / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data.qpos.copy(), data.qvel.copy()


def linearize(model, qpos_eq, qvel_eq):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_eq
    data.qvel[:] = qvel_eq
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    nv, nu = model.nv, model.nu
    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    return A, B


def solve_dare(A, B, Q, R, n_iter=2000, tol=1e-10):
    P = Q.copy()
    for _ in range(n_iter):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
    BtP = B.T @ P
    K = np.linalg.solve(R + BtP @ B, BtP @ A)
    return K


def build_cost_two_leg(model, shoulder_r_multiplier=500.0):
    """computer_graphics 프로젝트에서 300초 안정으로 검증된 두 발
    서있기용 Q/R 그대로. 이 프로젝트에서는 '이미 풀린 문제'의
    기준점(두 발 서있기)으로만 쓴다."""
    nv, nu = model.nv, model.nu
    Q = np.eye(2 * nv) * 0.01
    Q[0:3, 0:3] = np.eye(3) * 5.0
    Q[2, 2] = 2.0
    Q[3:6, 3:6] = np.eye(3) * 100.0
    Q[nv:nv + 3, nv:nv + 3] = np.eye(3) * 0.5
    Q[nv + 3:nv + 6, nv + 3:nv + 6] = np.eye(3) * 10.0
    R = np.eye(nu) * 0.2
    for i in range(nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name is not None and "shoulder" in name:
            R[i, i] = 0.2 * shoulder_r_multiplier
    return Q, R


def tilt_deg(qpos):
    """몸통(freejoint) 쿼터니언에서 업라이트 기준 기울기(도). qpos[4],
    qpos[5]는 쿼터니언의 x,y 성분 - 롤/피치가 합쳐진 전체 기울기량."""
    xq, yq = qpos[4], qpos[5]
    r33 = max(-1.0, min(1.0, 1 - 2 * (xq * xq + yq * yq)))
    return math.degrees(math.acos(r33))


def contact_zmp(model, data):
    """실제 접촉력(수직 성분)으로 가중 평균한 접촉점 위치 = ZMP.
    운동학적 가속도를 유한차분으로 추정하는 방식은 노이즈가 너무
    커서(2차 미분) 버렸다 - 접촉력 직접 사용이 훨씬 깨끗하다
    (정지 서있기에서 검증: 양발 중앙 근처로 거의 정확히 나옴).

    주의(이번 프로젝트에서 발견한 함정): 다리를 "양쪽 다 같은 방향"
    으로 명령하면 체중 이동이 안 되고 골반이 엉뚱하게 반응한다 -
    체중을 한쪽으로 옮기려면 스탠스(지지) 다리 쪽만 조정해야 한다.
    그리고 다리를 급하게 꺾으면 정적 캘리브레이션과 반대 방향으로
    ZMP가 순간적으로 튀는 전이(transient) 현상이 있었다 - 빠른
    피드백 루프를 그대로 믿지 말고 이 전이 효과를 감안해야 한다.
    """
    total_fz = 0.0
    wx = wy = 0.0
    force = np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, force)
        frame = data.contact[i].frame.reshape(3, 3)
        fz_world = (frame.T @ force[:3])[2]
        if fz_world <= 1e-6:
            continue
        pos = data.contact[i].pos
        total_fz += fz_world
        wx += fz_world * pos[0]
        wy += fz_world * pos[1]
    if total_fz < 1e-6:
        return 0.0, 0.0
    return wx / total_fz, wy / total_fz
