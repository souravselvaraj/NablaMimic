"""Gates 1 and 2 on real clips: can the feedforward replay a reference, and can the iLQR teacher hold it?

  python -m aasaan.teach --clips walk1_subject1 dance1_subject1 --start 5 --seconds 8 \\
      --refs runs/refs --out runs/teacher

CPU only. Per clip it takes a window of the smoothed reference and reports:
  r_cone_p95     how infeasible the window is to begin with (body weights)
  seat_mm        how far the reference had to be moved vertically for the simulator's contact to carry its load
                 (median and largest size): how far off the floor the clip's feet are
  defect_*       gate 1: one-period defect of the feedforward setpoints in the real simulator, from seated states
  open_loop_s    seconds the feedforward alone survives the benchmark's termination test
  start          which first pass gave a real trajectory: seated, bootstrap, or none
  held_s         seconds of the window the teacher's nominal covers (all of it when feasible)
  track_*        the nominal's largest error against the reference
  pushes         for each impulse (N s, applied to the torso over 0.1 s, a third of the way in): the directions
                 (forward, backward, sideways) the teacher survived
The teacher of each clip is saved to <out>/<clip>.npz (float32: q, v, u, K, P; k0 and the window in config.json).
--refs keeps the smoothed references on disk; without it every run refits the whole clip (about an hour each).
Linearization costs about 2.5 CPU-seconds per second of motion per pass: three passes by default, four when the
bootstrap is needed.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .exp0 import _clips, _eps, _print_table, _write
from .feasibility import Analyzer
from .model import SUBSTEPS
from .reference import EPS_DEFAULT, Reference, load_reference
from .teacher import Simulator, feedforward, ilqr, one_period_defect, open_loop, rollout, seat

_CTX = None
DIRS = {"fwd": (1.0, 0.0, 0.0), "back": (-1.0, 0.0, 0.0), "side": (0.0, 1.0, 0.0)}


def window(ref, start, seconds):
    """The part of a Reference between two times, cut at control steps."""
    a = int(round(start / ref.h / SUBSTEPS)) * SUBSTEPS
    b = min(len(ref.qpos), a + int(round(seconds / ref.h / SUBSTEPS)) * SUBSTEPS + 2)
    if b - a < 10 * SUBSTEPS:
        raise ValueError(f"{ref.name}: window {start}..{start + seconds} s leaves fewer than 10 control steps")
    return Reference(ref.name, ref.h, ref.qpos[a:b], ref.qvel[a:b], ref.qacc[a:b], ref.z_shift, ref.fit_rms)


def _one(job):
    global _CTX
    path, eps, start, seconds, passes, impulses, out, refs = job
    if _CTX is None:
        _CTX = (Analyzer(), Simulator())
    an, sim = _CTX
    t0 = time.time()
    ref = window(load_reference(an.m, path, eps, cache=refs), start, seconds)
    ff = feedforward(an, ref)
    qpos, shift = seat(an, ref)
    de = one_period_defect(sim, ref, ff, qpos)
    ol = rollout(sim, ref, open_loop(ref, ff, qpos=qpos), ff=ff)
    teacher = ilqr(sim, ref, ff, qpos, passes=passes)
    cl = rollout(sim, ref, teacher, ff=ff)
    n = len(teacher.u)
    row = {"clip": ref.name, "seconds": (min(len(ff.u), len(qpos) - 1) - 1) * 0.02,
           "r_cone_p95": float(np.percentile(ff.r_cone, 95)), "tau_ratio_max": float(ff.tau_ratio.max()),
           "seat_mm": float(np.median(np.abs(shift[1:])) * 1e3), "seat_mm_max": float(np.abs(shift[1:]).max() * 1e3),
           "defect_root_mm": float(np.median(de[:, 0]) * 1e3), "defect_root_mm_max": float(de[:, 0].max() * 1e3),
           "defect_joint_mrad": float(np.median(de[:, 2]) * 1e3),
           "open_loop_s": len(ol.u) * 0.02,
           "start": teacher.start if teacher.feasible else "none", "held_s": n * 0.02,
           "track_root_mm": float(cl.err[:, 0].max() * 1e3), "track_rot_mrad": float(cl.err[:, 1].max() * 1e3),
           "track_joint_mrad": float(cl.err[:, 2].max() * 1e3),
           "cost_first": teacher.cost[0] if teacher.cost else float("nan"),
           "cost_last": teacher.cost[-1] if teacher.cost else float("nan")}
    k_push = teacher.k0 + n // 3
    for imp in impulses:
        ok = []
        if teacher.feasible:
            for name, dirv in DIRS.items():
                push = {k: np.array(dirv) * imp / 0.1 for k in range(k_push, k_push + 5)}
                if rollout(sim, ref, teacher, push=push, ff=ff).fell < 0:
                    ok.append(name)
        row[f"push_{imp:g}"] = "+".join(ok) or "none"
    if out is not None and teacher.feasible:
        np.savez_compressed(Path(out) / f"{ref.name}.npz", k0=teacher.k0, start=start,
                            **{k: getattr(teacher, k).astype(np.float32) for k in ("q", "v", "u", "K", "P")})
    row["cpu_s"] = time.time() - t0
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aasaan.teach", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="directory of the LAFAN1 G1 CSVs (default $LAFAN1_G1_DIR)")
    ap.add_argument("--clips", nargs="+", default=None, help="clip names (default: every CSV)")
    ap.add_argument("--start", type=float, default=2.0, help="window start within each clip (s)")
    ap.add_argument("--seconds", type=float, default=8.0, help="window length (s)")
    ap.add_argument("--passes", type=int, default=1, help="iLQR passes after the first trajectory")
    ap.add_argument("--impulses", type=float, nargs="+", default=[5.0, 10.0, 20.0, 40.0], help="push sizes (N s)")
    ap.add_argument("--eps-pos", type=float, default=EPS_DEFAULT["pos"])
    ap.add_argument("--eps-quat", type=float, default=EPS_DEFAULT["quat"])
    ap.add_argument("--eps-joint", type=float, default=EPS_DEFAULT["joint"])
    ap.add_argument("--refs", default=None, help="directory that keeps the smoothed references between runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args(argv)
    paths = _clips(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), _eps(args), args.start, args.seconds, args.passes, args.impulses, str(out), args.refs)
            for p in paths]
    workers = max(1, min(args.workers, len(jobs)))
    if workers == 1:
        rows = [_one(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            rows = pool.map(_one, jobs, chunksize=1)
    _write(out / "gates.csv", rows)
    with open(out / "config.json", "w") as f:
        json.dump({"eps": _eps(args), "start": args.start, "seconds": args.seconds, "passes": args.passes,
                   "impulses": args.impulses, "clips": [p.stem for p in paths]}, f, indent=1)
    _print_table(rows, ("clip", "r_cone_p95", "seat_mm", "seat_mm_max", "defect_root_mm", "defect_joint_mrad",
                        "open_loop_s", "start", "held_s", "track_root_mm", "track_joint_mrad"))
    print("pushes survived (N s -> directions): " + "; ".join(
        f"{r['clip']}: " + ", ".join(f"{i:g} -> {r[f'push_{i:g}']}" for i in args.impulses) for r in rows))


if __name__ == "__main__":
    main()
