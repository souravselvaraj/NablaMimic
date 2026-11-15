"""Per-frame tracking error of a trained run, as the table `python -m aasaan.exp0 join` reads. Needs JAX and a GPU.

NOT YET RUN: written against nablamimic.evaluate (same environment settings and rollout), but the machine it was
written on has no GPU. Check its per-clip mean against evaluate's per_clip return before trusting the table.

Every clip is played once from its first frame exactly as the nominal evaluation does (deterministic action, no
randomization or noise, MuJoCo Warp). For each control step while the robot is up:

    error = 1 - (sum of the six tracking reward terms) / (their maximum)       in [0, 1]

  python -m aasaan.dump_tracking --run runs/base_s0 --out runs/exp0/tracking.csv [--iter 9000]

Columns: clip, frame, error. frame is the 50 Hz reference frame id within the clip that the robot is tracking after
the step.
"""

import argparse
import csv
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nablamimic import jax_cache, sim
from nablamimic.env import BM_BODIES, W, EnvConfig, TrackingEnv
from nablamimic.motion import all_clip_names, load_library
from nablamimic.nets import actor_apply
from nablamimic.sim import vmapped

TRACK = tuple(t for t in W if W[t] > 0)  # the tracking terms; the penalties have negative weights


def make_rollout(env, steps):
    @jax.jit
    def run(actor, frames, key, tables):
        f = vmapped(env, tables)
        n = frames.shape[0]
        k0, k1 = jax.random.split(key)
        state = f["reset"](jax.random.split(k0, n), frames)

        def body(c, xs):
            _, k = xs
            state, alive = c
            hist, goal = f["policy_obs"](state, jax.random.split(k, n))
            a = actor_apply(actor, hist, goal)
            s2 = f["step"](state, a)
            _, terms = f["reward"](s2, a, state.prev_a, prev=state)
            err = 1.0 - sum(terms[nm] for nm in TRACK) / env.max_reward
            out = (s2.frame, err, alive)
            alive = alive & ~f["terminated"](s2) & ~f["clip_ended"](s2)
            return (s2, alive), out

        _, (frame, err, alive) = jax.lax.scan(
            body, (state, jnp.ones(n, bool)), (jnp.arange(steps), jax.random.split(k1, steps)))
        return frame, err, alive  # each (steps, n)

    return run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aasaan.dump_tracking", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (config.json, ckpt/)")
    ap.add_argument("--iter", type=int, default=None, help="checkpoint iteration (default: the last one)")
    ap.add_argument("--out", required=True, help="CSV to write")
    ap.add_argument("--solver-iterations", type=int, default=10)
    args = ap.parse_args(argv)
    jax_cache.enable()
    jax.config.update("jax_default_matmul_precision", "highest")
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    names = all_clip_names() if cfg["clips"] == ["all"] else cfg["clips"]
    lib = load_library(names, bodies=BM_BODIES)
    env = TrackingEnv(lib, EnvConfig(solver_iterations=args.solver_iterations, dr=False, obs_noise=False,
                                     reset_noise=False, residual_action=bool(cfg.get("residual_action", False)),
                                     action_rate_space="setpoint"))
    nclip = len(env.starts)
    env.warp = sim.make_warp(env, [nclip], train=False)
    its = sorted(int(p.stem[3:]) for p in (run / "ckpt").glob("it_*.pkl"))
    it = args.iter if args.iter is not None else its[-1]
    with open(run / "ckpt" / f"it_{it:06d}.pkl", "rb") as f:
        ck = pickle.load(f)
    actor = jax.tree_util.tree_map(jnp.asarray, ck["actor_ema" if "actor_ema" in ck else "actor"])
    starts = env.starts.astype(np.int32)
    fn = make_rollout(env, int(env.lengths.max() - 1))
    frame, err, alive = (np.asarray(x) for x in jax.device_get(
        fn(actor, jnp.asarray(starts), jax.random.PRNGKey(1), env.tables())))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "frame", "error"])
        for c, name in enumerate(names):
            for t in np.nonzero(alive[:, c])[0]:
                w.writerow([name, int(frame[t, c] - starts[c]), f"{err[t, c]:.6f}"])
    kept = alive.sum(0)
    print(f"checkpoint {it}: {int(kept.sum())} rows over {nclip} clips -> {out}")
    print("mean tracking error per clip: " + ", ".join(
        f"{n} {err[:, c][alive[:, c]].mean():.3f}" for c, n in enumerate(names[:5])) + (" ..." if nclip > 5 else ""))


if __name__ == "__main__":
    main()
