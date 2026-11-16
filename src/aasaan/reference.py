"""Reference clips for inverse dynamics: smooth in time, sampled at the physics rate, with discrete derivatives.

The benchmark builds its 50 Hz references by straight-line interpolation between the 30 FPS frames
(nablamimic.motion._resample). That is fine for a tracking target but useless for inverse dynamics: acceleration is
zero between source frames and a spike at each one. Here every qpos channel gets a quintic smoothing spline and is
sampled at 200 Hz (the physics rate), so sample 4k is the benchmark's 50 Hz frame k.

Velocity and acceleration follow the simulator's own integrator, which is what makes a simulated trajectory come out
with zero residual (tests/test_aasaan_exp0.py):

    v[t] = (q[t] - q[t-1]) / h        (mj_differentiatePos, so quaternions are handled)
    a[t] = (v[t+1] - v[t]) / h
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import make_interp_spline, make_splrep

from nablamimic import motion

from .model import TIMESTEP, scene_path

FPS_IN = 30.0
# RMS distance each smoothing spline may sit from the source frames, per channel kind
EPS_DEFAULT = dict(pos=0.002, quat=0.002, joint=0.003)  # m, quaternion component, rad


@dataclass
class Reference:
    name: str
    h: float            # sample period
    qpos: np.ndarray    # (T, nq)
    qvel: np.ndarray    # (T, nv)  valid for t >= 1
    qacc: np.ndarray    # (T, nv)  valid for 1 <= t <= T - 2
    z_shift: float      # floor alignment subtracted from root z
    fit_rms: dict       # achieved RMS distance from the source frames, per channel kind

    @property
    def valid(self):
        """Frames whose position, velocity and acceleration are all defined."""
        return slice(1, len(self.qpos) - 1)


def load_csv(path):
    """Rows at 30 FPS: root xyz, root quaternion (x, y, z, w), the 29 joints -> (N, 36) with the quaternion as wxyz."""
    raw = np.loadtxt(path, delimiter=",")
    return to_wxyz(raw)


def to_wxyz(raw):
    return np.concatenate([raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 7:]], axis=1)


def to_csv_rows(qpos):
    """Inverse of load_csv's reordering (for writing test clips)."""
    return np.concatenate([qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]], axis=1)


def floor_shift(raw_wxyz):
    """The benchmark's floor alignment (nablamimic.motion.process_clip): the 5th percentile of the lowest foot point
    of the 50 Hz reference, on Menagerie's own model. Computed the same way so both references share one floor."""
    m = mujoco.MjModel.from_xml_path(str(scene_path()))
    d = mujoco.MjData(m)
    qpos = motion._resample(to_csv_rows(raw_wxyz), FPS_IN, 50.0)
    low = np.empty(len(qpos))
    for i, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        low[i] = motion._foot_min_z(m, d)
    return float(np.percentile(low, 5))


def _unwrap_quat(q):
    """Flip signs so consecutive quaternions lie in the same hemisphere (a spline through raw signs would tumble)."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def _fit(t, y, eps, k=5):
    if eps <= 0 or len(t) <= k + 1:
        return make_interp_spline(t, y, k=min(k, len(t) - 1))
    return make_splrep(t, y, k=k, s=len(t) * eps ** 2)


def sample_times(n_in, rate):
    """Same span as the benchmark's resampler, so sample (rate / 50) * k is its 50 Hz frame k."""
    return np.arange(0.0, (n_in - 1) / FPS_IN, 1.0 / rate)


def smooth_qpos(raw_wxyz, rate=1.0 / TIMESTEP, eps=None, kind="spline"):
    """(qpos at `rate`, achieved fit RMS per channel kind).

    kind="spline": quintic smoothing splines, each allowed an RMS distance eps[kind] from the source frames
    (0 = interpolate them exactly). kind="linear": the benchmark's straight-line / slerp resampling, for comparison.
    """
    eps = {**EPS_DEFAULT, **(eps or {})}
    n = len(raw_wxyz)
    t_in = np.arange(n) / FPS_IN
    t_out = sample_times(n, rate)
    if kind == "linear":
        return motion._resample(to_csv_rows(raw_wxyz), FPS_IN, rate), dict(pos=0.0, quat=0.0, joint=0.0)
    if kind != "spline":
        raise ValueError(f"kind must be 'spline' or 'linear', got {kind!r}")
    src = raw_wxyz.copy()
    src[:, 3:7] = _unwrap_quat(src[:, 3:7])
    kinds = ["pos"] * 3 + ["quat"] * 4 + ["joint"] * (src.shape[1] - 7)
    out = np.empty((len(t_out), src.shape[1]))
    sq = dict(pos=[], quat=[], joint=[])
    for c, ch in enumerate(kinds):
        spl = _fit(t_in, src[:, c], eps[ch])
        out[:, c] = spl(t_out)
        sq[ch].append(np.mean((spl(t_in) - src[:, c]) ** 2))
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out, {k: float(np.sqrt(np.mean(v))) for k, v in sq.items()}


def kinematics(m, qpos, h):
    """Discrete velocity and acceleration of a position sequence, in the simulator's convention (module docstring)."""
    T = len(qpos)
    qvel = np.zeros((T, m.nv))
    for t in range(1, T):
        mujoco.mj_differentiatePos(m, qvel[t], h, qpos[t - 1], qpos[t])
    qacc = np.zeros((T, m.nv))
    qacc[1:T - 1] = (qvel[2:] - qvel[1:T - 1]) / h
    return qvel, qacc


def build_reference(m, raw_wxyz, name="clip", eps=None, kind="spline", floor_align=True):
    """Smoothed reference of one clip at the model's timestep, floor-aligned like the benchmark's."""
    h = float(m.opt.timestep)
    qpos, fit = smooth_qpos(raw_wxyz, 1.0 / h, eps, kind)
    z = floor_shift(raw_wxyz) if floor_align else 0.0
    qpos[:, 2] -= z
    qvel, qacc = kinematics(m, qpos, h)
    return Reference(name, h, qpos, qvel, qacc, z, fit)


def load_reference(m, path, eps=None, kind="spline", cache=None):
    """build_reference for one CSV. With `cache` (a directory) the smoothed positions are stored there once per
    clip and smoothing setting and read back afterwards: fitting the splines of a full clip takes about an hour of
    CPU, everything after it seconds. The cache holds motion data, so it belongs with the dataset, never in git."""
    path = Path(path)
    eps = {**EPS_DEFAULT, **(eps or {})}
    raw = load_csv(path)
    if cache is None:
        return build_reference(m, raw, path.stem, eps, kind)
    tag = kind if kind == "linear" else "spline_" + "_".join(f"{eps[k]:g}" for k in ("pos", "quat", "joint"))
    f = Path(cache) / f"{path.stem}__{tag}.npz"
    h = float(m.opt.timestep)
    if f.exists():
        z = np.load(f)
        if int(z["frames"]) == len(raw) and float(z["h"]) == h:
            qvel, qacc = kinematics(m, z["qpos"], h)
            return Reference(path.stem, h, z["qpos"], qvel, qacc, float(z["z_shift"]), json.loads(str(z["fit_rms"])))
    ref = build_reference(m, raw, path.stem, eps, kind)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npz")  # written whole, then renamed: workers may race
    np.savez(tmp, qpos=ref.qpos, z_shift=ref.z_shift, fit_rms=json.dumps(ref.fit_rms), frames=len(raw), h=h)
    os.replace(tmp, f)
    return ref
