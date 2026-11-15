"""The benchmark's G1 on CPU MuJoCo, without JAX.

build_model() repeats nablamimic.env.build_model (same robot.build, timestep, solver iterations and PD gains); it is
duplicated here only because nablamimic.env imports JAX at module level. tests/test_aasaan_exp0.py checks the two
stay in step wherever nablamimic.env can be imported.
"""

import os
from pathlib import Path

import mujoco
import numpy as np

from nablamimic import robot

ROOT = Path(__file__).resolve().parents[2]
SUBSTEPS, TIMESTEP, DT = 4, 0.005, 0.02  # 4 x 5 ms physics steps per 50 Hz control period
OMEGA = 10 * 2.0 * np.pi  # mjlab's actuator natural frequency
NBASE = 6  # unactuated floating-base rows of every generalized-force vector


def menagerie_dir():
    return Path(os.environ.get("MENAGERIE_DIR") or ROOT / "third_party/mujoco_menagerie")


def scene_path():
    return menagerie_dir() / "unitree_g1/scene_mjx.xml"


def build_model(solver_iterations=10, discrete_inverse=False, contacts=True):
    """(model, kp, kd) of the benchmark G1.

    discrete_inverse: mj_inverse reads qacc as the discrete acceleration (v[t+1] - v[t]) / h of the model's own
    integrator (implicitfast), so a trajectory the simulator produced has zero inverse-dynamics residual.
    contacts=False disables contact forces (joint limits stay): the applied force a motion needs with no ground.
    """
    scene = scene_path()
    if not scene.exists():
        raise FileNotFoundError(f"{scene} not found: run scripts/setup_menagerie.sh (or set MENAGERIE_DIR)")
    m = robot.build(scene)
    m.opt.iterations = solver_iterations
    m.opt.timestep = TIMESTEP
    kp, kd = np.empty(m.nu), np.empty(m.nu)
    for i in range(m.nu):
        dof = m.jnt_dofadr[m.actuator_trnid[i, 0]]
        arm = m.dof_armature[dof]
        kp[i] = arm * OMEGA ** 2
        kd[i] = 2.0 * 2.0 * arm * OMEGA
        m.actuator_gainprm[i, 0] = kp[i]
        m.actuator_biasprm[i, 1] = -kp[i]
        m.actuator_biasprm[i, 2] = -kd[i]
    robot.widen_ctrlrange(m, kp)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)  # sensors are never read
    if discrete_inverse:
        m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
    if not contacts:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    return m, kp, kd


def torque_limits(m):
    """Per-actuator torque limit (the joint's actuator force range, as the benchmark clamps it)."""
    return np.array([m.jnt_actfrcrange[m.actuator_trnid[i, 0], 1] for i in range(m.nu)])


def implied_setpoint(tau, q, v, kp, kd):
    """The joint setpoint whose PD torque kp * (u - q) - kd * v equals tau (valid while tau is inside the limits)."""
    return q + (tau + kd * v) / kp


def weight(m):
    """m * g of the whole robot, the force scale of the normalized residuals."""
    return float(mujoco.mj_getTotalmass(m) * np.linalg.norm(m.opt.gravity))
