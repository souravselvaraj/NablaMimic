"""Kinematic test motions that need no dataset: a squat with arm swing, feet planted.

Joint angles are prescribed and the floating base is solved so the left foot never moves (the right follows by
symmetry). Nothing here enforces dynamics, and the soles are only flat to about a millimetre, so it is a fair
stand-in for a retargeted clip: kinematically plausible, not a trajectory of the simulator.
"""

import mujoco
import numpy as np

from .model import build_model
from .reference import Reference, kinematics


def soles_on_floor(m, qpos, depth=0.0003):
    """qpos shifted in z so its lowest floor point sits `depth` below the floor."""
    geo, _, _ = build_model()
    floor = geo.geom("floor").id
    for p in range(geo.npair):
        if floor in (geo.pair_geom1[p], geo.pair_geom2[p]):
            geo.pair_margin[p] = 2.0
    d = mujoco.MjData(geo)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(geo, d)
    mujoco.mj_collision(geo, d)
    q = qpos.copy()
    q[2] -= min(d.contact[i].dist for i in range(d.ncon) if floor in (d.contact[i].geom1, d.contact[i].geom2)) + depth
    return q


def squat(m, seconds=3.0, freq=0.5, amp=0.15, arm=0.3, ramp=0.5):
    """Reference of a squat at `freq` Hz with hip/knee/ankle amplitude `amp` rad and shoulder swing `arm` rad,
    eased in over `ramp` seconds, at the model's timestep."""
    h = float(m.opt.timestep)
    d = mujoco.MjData(m)
    q0 = soles_on_floor(m, m.key("knees_bent").qpos.copy())
    names = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
    foot = m.body("left_ankle_roll_link").id

    def idx(sub):
        return [i for i, n in enumerate(names) if sub in n]

    def foot_pose(q):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        return d.xpos[foot].copy(), d.xmat[foot].reshape(3, 3).copy()

    p0, r0 = foot_pose(q0)
    T = int(round(seconds / h)) + 1
    qpos, quat = np.zeros((T, m.nq)), np.zeros(4)
    for t in range(T):
        r = min(1.0, t * h / ramp)
        s = np.sin(2 * np.pi * freq * t * h) * r ** 3 * (10 - 15 * r + 6 * r * r)
        j = q0[7:].copy()
        j[idx("hip_pitch")] -= amp * s
        j[idx("knee")] += 2 * amp * s
        j[idx("ankle_pitch")] -= amp * s
        j[idx("left_shoulder_pitch")] += arm * s
        j[idx("right_shoulder_pitch")] -= arm * s
        pf, rf = foot_pose(np.concatenate([[0, 0, 0, 1, 0, 0, 0], j]))
        rb = r0 @ rf.T
        mujoco.mju_mat2Quat(quat, rb.ravel())
        qpos[t] = np.concatenate([p0 - rb @ pf, quat, j])
    qvel, qacc = kinematics(m, qpos, h)
    return Reference("squat", h, qpos, qvel, qacc, 0.0, {})
