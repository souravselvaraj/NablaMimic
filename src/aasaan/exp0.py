"""Experiment 0: does reference infeasibility predict where trackers struggle? CPU only, no training.

  python -m aasaan.exp0 run   --out runs/exp0                 # every clip in $LAFAN1_G1_DIR (default data/lafan1_g1)
  python -m aasaan.exp0 sweep --clips walk1_subject1          # is the result stable as the smoothing changes?
  python -m aasaan.exp0 join  --exp0 runs/exp0 --tracking err.csv

run writes, under --out:
  summary.csv        one row per clip (percentiles of both residuals, flight fraction, torque-limit fractions)
  windows.csv        one row per clip and 1 s window, keyed by 50 Hz reference frame ids
  frames/<clip>.npz  the per-frame arrays at 200 Hz
join reads a tracking-error table (columns: clip, frame, error; frame = 50 Hz reference frame id within the clip,
one row per control step of a rollout) and reports how window residuals relate to window tracking error.
"""

import argparse
import csv
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .feasibility import Analyzer, summarize, windows
from .model import ROOT
from .reference import EPS_DEFAULT, build_reference, load_csv

_AN = None  # one Analyzer per worker process


def data_dir(arg=None):
    return Path(arg or os.environ.get("LAFAN1_G1_DIR") or ROOT / "data/lafan1_g1")


def _eps(args):
    return dict(pos=args.eps_pos, quat=args.eps_quat, joint=args.eps_joint)


def _one(job):
    global _AN
    path, eps, kind, tol, length, frames_dir = job
    if _AN is None or (_AN.tol, _AN.length) != (tol, length):
        _AN = Analyzer(tol=tol, length=length)
    t0 = time.time()
    ref = build_reference(_AN.m, load_csv(path), Path(path).stem, eps, kind)
    diag = _AN.analyze(ref)
    if frames_dir is not None:
        np.savez_compressed(Path(frames_dir) / f"{ref.name}.npz", h=ref.h, z_shift=ref.z_shift,
                            **{k: getattr(diag, k) for k in diag.__dataclass_fields__})
    row = {"clip": ref.name, **summarize(diag, ref.h), "z_shift": ref.z_shift,
           **{f"fit_rms_{k}": v for k, v in ref.fit_rms.items()}, "cpu_s": time.time() - t0}
    win = [{"clip": ref.name, **w} for w in windows(diag, ref.h)]
    return row, win


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _clips(args):
    d = data_dir(args.data)
    paths = sorted(d.glob("*.csv"))
    if args.clips:
        paths = [p for p in paths if p.stem in set(args.clips)]
        missing = set(args.clips) - {p.stem for p in paths}
        if missing:
            raise SystemExit(f"not in {d}: {sorted(missing)}")
    if not paths:
        raise SystemExit(f"no CSVs in {d} (scripts/download_lafan1.sh, or pass --data)")
    return paths


def run(args):
    paths = _clips(args)
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), _eps(args), args.kind, args.tol, args.length, str(out / "frames")) for p in paths]
    workers = max(1, min(args.workers, len(jobs)))
    t0 = time.time()
    if workers == 1:
        results = [_one(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            results = pool.map(_one, jobs, chunksize=1)
    rows = [r for r, _ in results]
    wins = [w for _, ws in results for w in ws]
    _write(out / "summary.csv", rows)
    _write(out / "windows.csv", wins)
    with open(out / "config.json", "w") as f:
        json.dump({"eps": _eps(args), "kind": args.kind, "tol": args.tol, "length": args.length,
                   "clips": [p.stem for p in paths]}, f, indent=1)
    print(f"{len(rows)} clips, {sum(r['frames'] for r in rows)} frames in {time.time() - t0:.0f} s -> {out}")
    _print_table(rows, ("clip", "seconds", "flight_frac", "r_soft_p50", "r_soft_p95", "r_cone_p50", "r_cone_p95",
                        "r_cone_max", "r_cone_frac_gt_0.1"))


def _print_table(rows, cols):
    width = max(len(str(r[cols[0]])) for r in rows)
    print(f"{cols[0]:<{width}}  " + "  ".join(f"{c:>18}" for c in cols[1:]))
    for r in rows:
        print(f"{str(r[cols[0]]):<{width}}  " + "  ".join(f"{r[c]:>18.4g}" for c in cols[1:]))


def sweep(args):
    """The diagnostic is only worth reading where it does not move with the smoothing: a residual that halves when eps
    doubles is measuring the differentiation of noisy source frames, not the motion."""
    paths = _clips(args)
    scales = [float(s) for s in args.scales]
    rows = []
    for s in scales:
        eps = {k: v * s for k, v in _eps(args).items()}
        for p in paths:
            row, _ = _one((str(p), eps, "spline", args.tol, args.length, None))
            rows.append({"clip": row["clip"], "eps_scale": s, **{k: row[k] for k in (
                "r_soft_p50", "r_soft_p95", "r_cone_p50", "r_cone_p95", "r_cone_max", "acc_p99", "fit_rms_joint")}})
    for p in paths:  # the benchmark's own straight-line resampling, as the reference point
        row, _ = _one((str(p), _eps(args), "linear", args.tol, args.length, None))
        rows.append({"clip": row["clip"], "eps_scale": float("nan"), **{k: row[k] for k in (
            "r_soft_p50", "r_soft_p95", "r_cone_p50", "r_cone_p95", "r_cone_max", "acc_p99", "fit_rms_joint")}})
    print("eps_scale nan = straight-line resampling (the benchmark's 50 Hz reference construction)")
    _print_table(rows, ("clip", "eps_scale", "r_soft_p50", "r_cone_p50", "r_cone_p95", "r_cone_max", "acc_p99",
                        "fit_rms_joint"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _write(args.out, rows)


def _rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):  # average ranks of ties
        sel = x == v
        if sel.sum() > 1:
            r[sel] = r[sel].mean()
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def join(args):
    """Window-level relation between reference residual and tracking error.

    Reports the rank correlation over all windows and within clips (a clip-level offset, e.g. a hard dance, would
    otherwise pass for a per-window effect), and the mean error in the window after a residual spike against the
    clip's other windows (does error rise just after the reference turns infeasible?)."""
    with open(Path(args.exp0) / "windows.csv") as f:
        wins = list(csv.DictReader(f))
    err = {}
    with open(args.tracking) as f:
        for r in csv.DictReader(f):
            err.setdefault(r["clip"], []).append((int(r["frame"]), float(r["error"])))
    rows = []
    for w in wins:
        e = [v for fr, v in err.get(w["clip"], ()) if int(w["frame0"]) <= fr < int(w["frame1"])]
        if e:
            rows.append((w["clip"], int(w["frame0"]), float(w[args.residual]), float(np.mean(e))))
    if not rows:
        raise SystemExit("no window has tracking rows: check the clip names and frame ids of --tracking")
    clips = sorted({r[0] for r in rows})
    res, e = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
    within = [spearman([r[2] for r in rows if r[0] == c], [r[3] for r in rows if r[0] == c]) for c in clips]
    within = np.array([w for w in within if np.isfinite(w)])
    print(f"{len(rows)} windows over {len(clips)} clips, residual = {args.residual}")
    print(f"rank correlation, all windows:      {spearman(res, e):+.3f}")
    if len(within):
        print(f"rank correlation, within clip:      median {np.median(within):+.3f}, "
              f"positive in {int((within > 0).sum())} of {len(within)} clips")
    thr = np.percentile(res, args.spike_percentile)
    after, rest = [], []
    for c in clips:
        cr = sorted((r for r in rows if r[0] == c), key=lambda r: r[1])
        spike = [i for i, r in enumerate(cr) if r[2] >= thr]
        nxt = {i + 1 for i in spike if i + 1 < len(cr)}
        after += [cr[i][3] for i in nxt]
        rest += [r[3] for i, r in enumerate(cr) if i not in nxt and i not in spike]
    if after and rest:
        print(f"mean error in the window after a spike (top {100 - args.spike_percentile:g}%): {np.mean(after):.4g} "
              f"over {len(after)} windows; other non-spike windows: {np.mean(rest):.4g} over {len(rest)}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aasaan.exp0", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", default=None, help="directory of the LAFAN1 G1 CSVs (default $LAFAN1_G1_DIR)")
        p.add_argument("--clips", nargs="+", default=None, help="clip names (default: every CSV)")
        p.add_argument("--eps-pos", type=float, default=EPS_DEFAULT["pos"], help="spline RMS tolerance, root xyz (m)")
        p.add_argument("--eps-quat", type=float, default=EPS_DEFAULT["quat"], help="... root quaternion components")
        p.add_argument("--eps-joint", type=float, default=EPS_DEFAULT["joint"], help="... joint angles (rad)")
        p.add_argument("--tol", type=float, default=0.02, help="height (m) under which a point may carry floor force")
        p.add_argument("--length", type=float, default=0.75, help="length L (m) that scales torques: m*g*L")

    p = sub.add_parser("run", help="residuals of every clip")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--kind", default="spline", choices=["spline", "linear"])
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.set_defaults(fn=run)

    p = sub.add_parser("sweep", help="sensitivity to the smoothing")
    common(p)
    p.add_argument("--scales", nargs="+", default=["0", "0.5", "1", "2", "4"],
                   help="multiples of the eps defaults (0 = interpolate the source frames exactly)")
    p.add_argument("--out", default=None, help="optional CSV of the table")
    p.set_defaults(fn=sweep)

    p = sub.add_parser("join", help="relate window residuals to tracking error")
    p.add_argument("--exp0", required=True, help="--out directory of a run")
    p.add_argument("--tracking", required=True, help="CSV with columns clip, frame, error")
    p.add_argument("--residual", default="r_cone_max",
                   choices=["r_cone_max", "r_cone_mean", "r_soft_max", "r_soft_mean"])
    p.add_argument("--spike-percentile", type=float, default=95.0)
    p.set_defaults(fn=join)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
