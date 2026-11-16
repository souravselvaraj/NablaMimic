"""Can linear feedback hold a trajectory that is known to be feasible?

  python -m aasaan.hold --traj runs/hold/walk1_subject1.npz runs/hold/run1_subject2.npz --out runs/hold

The teacher gates (aasaan.teach) failed on every real clip, and that failure has two possible sources that this
separates: the references are too far from anything the simulator can do, or Riccati feedback at 50 Hz cannot hold
a walking robot through its contact changes at all. Here the nominal is not built from a reference. It is a rollout
that a working tracker already produced (for example the baseline policy in the benchmark's own simulator), so
feasibility is given and only the feedback is on trial.

Each --traj file is an .npz with, at the 50 Hz control rate,
  qpos (N + 1, 36)   x y z, quaternion (w x y z), 29 joint angles, in the model's order
  qvel (N + 1, 35)   MuJoCo convention: linear velocity in the world frame, angular in the body frame, joint rates
  ctrl (N, 29)       the PD position targets the simulator received during period k (after any clipping), such that
                     state k + 1 is state k advanced by four 5 ms physics steps under ctrl[k]

Reported per trajectory (CPU MuJoCo throughout):
  kin_*        does the file agree with itself? Position differences over each period against the logged
               velocities, as a fraction of the typical speed, for the root's linear and angular velocity and the
               joints. A correct log gives a few percent; a wrong quaternion order or velocity frame gives about 1.
               Blank where the trajectory barely moves that way.
  engine_*     one-period defect of the logged (state, target) pairs: how well this simulator reproduces the one
               that made the log. A float32 copy of a log this simulator made gives 2e-5 mm. Above 0.02 mm or
               0.2 mrad a WARNING is printed: either the file's conventions are off (kin_* will say) or the two
               simulators really differ, and then the hold below is a test of both at once.
  replay_s     seconds the logged targets alone, with no feedback, stay inside the benchmark's termination test
  start        log = feedback linearized along the logged states held the first rollout; bootstrap = it needed
               gains linearized at the simulator's own state; none = neither held
  log_s        seconds the first of those lasted
  held_s       seconds held (all of it unless start is none)
  track_*      largest error of the held rollout against the logged trajectory
  push_<I>     for an impulse of I N s on the torso over 0.1 s, a third of the way in, per direction:
               R recovered (root back within 3 cm of the log at the end), S still up but further off, F fell
"""

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from .exp0 import _print_table, _write
from .teacher import ControlPath, Setpoints, Simulator, ilqr, one_period_defect, open_loop, rollout

_SIM = None
DIRS = {"fwd": (1.0, 0.0, 0.0), "back": (-1.0, 0.0, 0.0), "side": (0.0, 1.0, 0.0)}
RECOVERED = 0.03  # m


def load(path, sim):
    z = np.load(path)
    q, v, u = (np.asarray(z[k], float) for k in ("qpos", "qvel", "ctrl"))
    if q.shape[1:] != (sim.m.nq,) or v.shape != (len(q), sim.nv) or u.shape != (len(q) - 1, sim.nu):
        raise ValueError(f"{path}: expected qpos (N+1, {sim.m.nq}), qvel (N+1, {sim.nv}), ctrl (N, {sim.nu}); "
                         f"got {q.shape}, {v.shape}, {u.shape}")
    norm = np.linalg.norm(q[:, 3:7], axis=1)
    if np.abs(norm - 1).max() > 1e-3:
        raise ValueError(f"{path}: columns 3..6 of qpos are not unit quaternions "
                         f"(norm {norm.min():.3f}..{norm.max():.3f})")
    q[:, 3:7] /= norm[:, None]
    return ControlPath(Path(path).stem, q, v), Setpoints(u)


def kinematic_check(sim, path, floors=(0.05, 0.1, 0.1)):
    """Median mismatch between (q[k+1] - q[k]) / dt and the mean of the logged velocities at both ends, relative to
    the median speed, for root linear, root angular and joint velocities. nan where the median speed is under the
    floor (m/s, rad/s, rad/s): too little motion to tell."""
    nv, dq = sim.nv, np.zeros(sim.nv)
    diff, mean = [], []
    for k in range(len(path.qpos) - 1):
        mujoco.mj_differentiatePos(sim.m, dq, 0.02, path.qpos[k], path.qpos[k + 1])
        diff.append(dq.copy()), mean.append(0.5 * (path.qvel[k] + path.qvel[k + 1]))
    diff, mean = np.array(diff), np.array(mean)
    out = []
    for sl, floor in zip((slice(0, 3), slice(3, 6), slice(6, nv)), floors):
        speed = np.median(np.abs(mean[:, sl]).max(1))
        gap = np.median(np.abs(diff[:, sl] - mean[:, sl]).max(1))
        out.append(float(gap / speed) if speed > floor else float("nan"))
    return out


def hold(sim, path, ff, passes=1, impulses=(), out=None, tol=None):
    """One row of the report for a logged trajectory."""
    t0 = time.time()
    n = len(ff.u)
    de = one_period_defect(sim, path, ff, k0=0)
    ol = rollout(sim, path, open_loop(path, ff, k0=0), ff=ff, tol=tol)
    teacher = ilqr(sim, path, ff, k0=0, passes=passes, tol=tol)
    cl = rollout(sim, path, teacher, ff=ff, tol=tol)
    kin = kinematic_check(sim, path)
    row = {"clip": path.name, "seconds": n * 0.02, "kin_lin": kin[0], "kin_ang": kin[1], "kin_joint": kin[2],
           "engine_root_mm": float(np.median(de[:, 0]) * 1e3), "engine_root_mm_max": float(de[:, 0].max() * 1e3),
           "engine_joint_mrad": float(np.median(de[:, 2]) * 1e3),
           "engine_joint_mrad_max": float(de[:, 2].max() * 1e3), "engine_vel": float(np.median(de[:, 3])),
           "replay_s": len(ol.u) * 0.02,
           "start": {"seated": "log"}.get(teacher.start, teacher.start) if teacher.feasible else "none",
           "log_s": (teacher.seated_fell * 0.02 if teacher.seated_fell >= 0 else n * 0.02),
           "held_s": len(teacher.u) * 0.02,
           "track_root_mm": float(cl.err[:, 0].max() * 1e3), "track_rot_mrad": float(cl.err[:, 1].max() * 1e3),
           "track_joint_mrad": float(cl.err[:, 2].max() * 1e3),
           "gain_max": float(np.abs(teacher.K).max())}
    k_push = n // 3
    values = {}
    for imp in impulses:
        marks = []
        for name, dirv in DIRS.items():
            if not teacher.feasible:
                marks.append(f"{name}:-")
                continue
            push = {k: np.array(dirv) * imp / 0.1 for k in range(k_push, k_push + 5)}
            r = rollout(sim, path, teacher, push=push, ff=ff, tol=tol)
            mark = "F" if r.fell >= 0 else ("R" if r.err[-1, 0] < RECOVERED else "S")
            marks.append(f"{name}:{mark}")
            values[f"value_{imp:g}_{name}"] = r.value
        row[f"push_{imp:g}"] = " ".join(marks)
    if out is not None and teacher.feasible:
        np.savez_compressed(Path(out) / f"{path.name}_teacher.npz", k0=teacher.k0, **values,
                            **{k: getattr(teacher, k).astype(np.float32) for k in ("q", "v", "u", "K", "P")})
    row["cpu_s"] = time.time() - t0
    return row


def _one(job):
    global _SIM
    path, passes, impulses, out = job
    if _SIM is None:
        _SIM = Simulator()
    return hold(_SIM, *load(path, _SIM), passes=passes, impulses=impulses, out=out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aasaan.hold", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", nargs="+", required=True, help=".npz files with qpos, qvel, ctrl at 50 Hz")
    ap.add_argument("--passes", type=int, default=1, help="iLQR passes after the first trajectory")
    ap.add_argument("--impulses", type=float, nargs="+", default=[5.0, 10.0, 20.0, 40.0], help="push sizes (N s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), args.passes, args.impulses, str(out)) for p in args.traj]
    workers = max(1, min(args.workers, len(jobs)))
    if workers == 1:
        rows = [_one(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            rows = pool.map(_one, jobs, chunksize=1)
    _write(out / "hold.csv", rows)
    _print_table(rows, ("clip", "seconds", "kin_lin", "kin_ang", "kin_joint", "engine_root_mm", "engine_joint_mrad",
                        "replay_s", "start", "log_s", "held_s", "track_root_mm", "track_joint_mrad"))
    for r in rows:
        bad = [k for k in ("kin_lin", "kin_ang", "kin_joint") if r[k] > 0.2]
        if bad:
            print(f"WARNING {r['clip']}: the file disagrees with itself ({', '.join(bad)}): positions and velocities "
                  "do not describe the same motion. Check quaternion order (w x y z), the velocity frames and the "
                  "50 Hz sampling before reading anything else in its row.")
        elif r["engine_root_mm"] > 0.02 or r["engine_joint_mrad"] > 0.2:
            print(f"WARNING {r['clip']}: this simulator does not reproduce the log (one-period defect "
                  f"{r['engine_root_mm']:.3g} mm, {r['engine_joint_mrad']:.3g} mrad). If ctrl is the target the "
                  "simulator really received, the two simulators differ and the hold is a test of both at once.")
    for imp in args.impulses:
        print(f"push {imp:g} N s: " + "; ".join(f"{r['clip']}: {r[f'push_{imp:g}']}" for r in rows))


if __name__ == "__main__":
    main()
