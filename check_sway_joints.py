import math

import numpy as np
import mujoco

import common

model = common.load_model()
nv, nu = model.nv, model.nu
qpos_eq0, qvel_eq0 = common.find_equilibrium(model)
A0, B0 = common.linearize(model, qpos_eq0, qvel_eq0)
Q, R = common.build_cost_two_leg(model)
K0 = common.solve_dare(A0, B0, Q, R)
ctrl_limits = np.array([math.radians(30.0)] * nu)

a_abduct_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_L")
a_abduct_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_hip_abduct_R")

AMPLITUDE_DEG, PERIOD_S = 3.0, 1.0

data = mujoco.MjData(model)
data.qpos[:] = qpos_eq0
data.qvel[:] = qvel_eq0

max_dev = np.zeros(nu)  # 각 액추에이터가 qpos_eq0에서 얼마나 벗어났는지 최대값
qadr_by_act = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(nu)]

for tick in range(int(10.0 / 0.01)):
    bias = math.radians(AMPLITUDE_DEG * math.sin(2 * math.pi * data.time / PERIOD_S))
    for _ in range(2):
        dq = np.zeros(nv)
        mujoco.mj_differentiatePos(model, dq, 1.0, qpos_eq0, data.qpos)
        dv = data.qvel - qvel_eq0
        x = np.concatenate([dq, dv])
        u = -K0 @ x
        u[a_abduct_R] = min(bias, 0.0)
        u[a_abduct_L] = max(bias, 0.0)
        u = np.clip(u, -ctrl_limits, ctrl_limits)
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
    for i in range(nu):
        dev = abs(data.qpos[qadr_by_act[i]] - qpos_eq0[qadr_by_act[i]])
        max_dev[i] = max(max_dev[i], dev)

print("각 관절의 qpos_eq0 대비 최대 이탈(도), 큰 것부터:")
order = np.argsort(-max_dev)
for i in order[:12]:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  {name}: {math.degrees(max_dev[i]):.3f}도")
