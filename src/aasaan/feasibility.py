"""Experiment 0: how far is a kinematic reference from a motion the robot could physically perform?

The robot's floating base has no actuator, so in any real motion the six base rows of the inverse-dynamics force
vanish. Whatever is left there is force that would have to come from nowhere. Two residuals per frame, both divided
by the robot's weight m*g (torques by m*g*L):

  r_soft  base residual of MuJoCo's own inverse dynamics, with the benchmark's soft contact model evaluated on the
          reference exactly as given. This is what a positions-only projection has to drive to zero. It is very
          sensitive to foot height: a foot 3 mm above the floor carries no force at all.

  r_cone  the smallest base residual any contact forces could leave: forces are free inside the friction pyramid at
          every robot point within `tol` of the floor (non-negative least squares). It ignores millimetre height
          errors and asks only whether the motion is dynamically possible. In flight (no point near the floor) it is
          the plain ballistic check.

r_soft >> r_cone means the reference only needs its contacts cleaned up; a large r_cone means the motion itself
breaks Newton-Euler and no tracker can follow it exactly.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import nnls

from .model import NBASE, build_model, torque_limits, weight

TANGENTS = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
UP = np.array([0.0, 0.0, 1.0])


@dataclass
class Diagnostics:
    """Per-frame arrays over the valid frames of a Reference (its frame t = index + 1)."""
    r_soft: np.ndarray      # normalized base residual, soft contact as given
    r_cone: np.ndarray      # normalized base residual, best friction-pyramid forces
    base_soft: np.ndarray   # (n, 6) raw base rows [N, N, N, N m, N m, N m]
    base_need: np.ndarray   # (n, 6) base wrench the motion needs from the ground
    n_soft: np.ndarray      # active soft contacts with the floor
    n_cand: np.ndarray      # candidate floor points within tol (0 = flight)
    tau_ratio_soft: np.ndarray  # max |joint torque| / limit, soft contact
    tau_ratio_cone: np.ndarray  # same with the best pyramid forces
    acc_max: np.ndarray     # max |joint acceleration| (rad/s^2), a smoothing sanity signal

    @property
    def flight(self):
        return self.n_cand == 0


class Analyzer:
    """Holds the three model variants; analyze() is the whole experiment for one reference."""

    def __init__(self, tol=0.02, length=0.75, solver_iterations=10):
        self.tol, self.length = tol, length
        self.m, self.kp, self.kd = build_model(solver_iterations, discrete_inverse=True)
        self.m_free, _, _ = build_model(solver_iterations, discrete_inverse=True, contacts=False)
        # geometry-only copy: floor pairs report any point closer than tol (never used for forces)
        self.m_geo, _, _ = build_model(solver_iterations)
        self.floor = self.m.geom("floor").id
        for p in range(self.m_geo.npair):
            if self.floor in (self.m_geo.pair_geom1[p], self.m_geo.pair_geom2[p]):
                self.m_geo.pair_margin[p] = tol
        self.d, self.d_free, self.d_geo = (mujoco.MjData(x) for x in (self.m, self.m_free, self.m_geo))
        self.mg = weight(self.m)
        self.scale = np.array([1.0 / self.mg] * 3 + [1.0 / (self.mg * length)] * 3)
        self.tau_max = torque_limits(self.m)
        self._jacp = np.zeros((3, self.m.nv))

    def norm(self, base):
        """Normalized size of base rows: forces over m*g, torques over m*g*L."""
        return float(np.linalg.norm(np.asarray(base) * self.scale))

    def _candidates(self, qpos):
        """Columns (nv, 4k) of generalized force per unit pyramid-edge force at each floor point within tol."""
        m, d = self.m_geo, self.d_geo
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_collision(m, d)
        cols = []
        for i in range(d.ncon):
            c = d.contact[i]
            if self.floor not in (c.geom1, c.geom2):
                continue
            geom = c.geom2 if c.geom1 == self.floor else c.geom1
            mujoco.mj_jac(m, d, self._jacp, None, c.pos, int(m.geom_bodyid[geom]))
            mu = float(c.friction[0])
            cols.append(self._jacp.T @ (UP[None] + mu * TANGENTS).T)  # (nv, 4)
        return np.concatenate(cols, axis=1) if cols else np.zeros((m.nv, 0))

    def frame(self, qpos, qvel, qacc):
        """All diagnostics of one frame as a tuple in Diagnostics field order."""
        d, df = self.d, self.d_free
        d.qpos[:], d.qvel[:], d.qacc[:] = qpos, qvel, qacc
        mujoco.mj_inverse(self.m, d)
        f_soft = d.qfrc_inverse.copy()
        n_soft = sum(1 for i in range(d.ncon) if self.floor in (d.contact[i].geom1, d.contact[i].geom2))
        df.qpos[:], df.qvel[:], df.qacc[:] = qpos, qvel, qacc
        mujoco.mj_inverse(self.m_free, df)
        f_need = df.qfrc_inverse.copy()
        A = self._candidates(qpos)
        if A.shape[1]:
            lam, _ = nnls(A[:NBASE] * self.scale[:, None], f_need[:NBASE] * self.scale)
            f_cone = f_need - A @ lam
        else:
            f_cone = f_need
        return (self.norm(f_soft[:NBASE]), self.norm(f_cone[:NBASE]), f_soft[:NBASE], f_need[:NBASE], n_soft,
                A.shape[1] // 4, float(np.max(np.abs(f_soft[NBASE:]) / self.tau_max)),
                float(np.max(np.abs(f_cone[NBASE:]) / self.tau_max)), float(np.max(np.abs(qacc[NBASE:]))))

    def analyze(self, ref):
        v = ref.valid
        rows = [self.frame(q, w, a) for q, w, a in zip(ref.qpos[v], ref.qvel[v], ref.qacc[v])]
        cols = list(zip(*rows))
        return Diagnostics(np.array(cols[0]), np.array(cols[1]), np.array(cols[2]), np.array(cols[3]),
                           np.array(cols[4]), np.array(cols[5]), np.array(cols[6]), np.array(cols[7]),
                           np.array(cols[8]))


def summarize(diag, h):
    """One clip in numbers. Percentiles rather than means: one bad landing can be the whole story."""
    def stats(x, prefix):
        if len(x) == 0:
            return {f"{prefix}_{k}": float("nan") for k in ("p50", "p95", "max")}
        return {f"{prefix}_p50": float(np.median(x)), f"{prefix}_p95": float(np.percentile(x, 95)),
                f"{prefix}_max": float(np.max(x))}

    fl = diag.flight
    out = {"frames": len(diag.r_soft), "seconds": len(diag.r_soft) * h, "flight_frac": float(fl.mean())}
    out.update(stats(diag.r_soft, "r_soft"))
    out.update(stats(diag.r_cone, "r_cone"))
    out.update(stats(diag.r_cone[fl], "r_cone_flight"))
    out.update(stats(diag.r_cone[~fl], "r_cone_stance"))
    out["r_cone_frac_gt_0.1"] = float((diag.r_cone > 0.1).mean())
    out["r_cone_frac_gt_0.3"] = float((diag.r_cone > 0.3).mean())
    out["tau_over_limit_frac_soft"] = float((diag.tau_ratio_soft > 1.0).mean())
    out["tau_over_limit_frac_cone"] = float((diag.tau_ratio_cone > 1.0).mean())
    out["acc_p99"] = float(np.percentile(diag.acc_max, 99))
    return out


def windows(diag, h, seconds=1.0, control_dt=0.02):
    """Per-window rows (default 1 s, the benchmark's reset bins). frame0 / frame1 are 50 Hz reference frame ids
    within the clip, so rows join directly against per-frame tracking error from a rollout."""
    per = int(round(seconds / h))
    sub = int(round(control_dt / h))
    rows = []
    n = len(diag.r_soft)
    for a in range(0, n, per):
        b = min(a + per, n)
        s = slice(a, b)
        # diag index i is reference sample i + 1 at the physics rate
        rows.append({"frame0": (a + 1) // sub, "frame1": (b + 1) // sub,
                     "r_soft_mean": float(diag.r_soft[s].mean()), "r_soft_max": float(diag.r_soft[s].max()),
                     "r_cone_mean": float(diag.r_cone[s].mean()), "r_cone_max": float(diag.r_cone[s].max()),
                     "flight_frac": float(diag.flight[s].mean()),
                     "tau_ratio_cone_max": float(diag.tau_ratio_cone[s].max())})
    return rows
