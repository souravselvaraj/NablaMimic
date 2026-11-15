# Aasaan

Feedback-distilled trajectory optimization for humanoid motion tracking, built next to ∇Mimic so every number is
measured on the benchmark's own G1, timestep and contact model. Work in progress; nothing in `src/nablamimic` is
changed.

## Experiment 0: does reference infeasibility predict tracking difficulty?

No training and no GPU. For every 200 Hz frame of every clip it asks how much force would have to act on the robot's
unactuated base for the reference to be a real motion, in units of body weight:

| residual | contact forces | what a large value means |
|---|---|---|
| `r_soft` | MuJoCo's soft contact, evaluated on the reference as given | what a positions-only projection must remove |
| `r_cone` | the best forces any friction pyramid allows at points within 2 cm of the floor | the motion itself breaks Newton–Euler |

```bash
python -m aasaan.exp0 sweep --clips <a few clip names>     # first: is the result stable as the smoothing changes?
python -m aasaan.exp0 run --out runs/exp0                  # all 40 clips: about 12 minutes on one core, parallel over clips
python -m aasaan.dump_tracking --run runs/base_s0 --out runs/exp0/tracking.csv   # GPU; after the baseline exists
python -m aasaan.exp0 join --exp0 runs/exp0 --tracking runs/exp0/tracking.csv
```

Read the `sweep` table before anything else. Where `r_cone` halves when the smoothing doubles, it is measuring the
differentiation of noisy 30 FPS frames, not the motion; trust only the range where it is flat.

How to read `join`:

| finding | implication |
|---|---|
| residual predicts tracking error within clips | projection addresses a real source of difficulty |
| spikes cluster in flight and at touchdown | contact handling comes first |
| high error where the residual is low | the limit is feedback or the policy, not the reference |
| low error where the residual is high | the tracker already compensates; projection buys little |

## What has been measured so far

On CPU MuJoCo 3.17 with the benchmark model (`tests/test_aasaan_exp0.py`):

- **The convention is exact.** With `mjENBL_INVDISCRETE`, `v[t] = (q[t] - q[t-1]) / h` and
  `a[t] = (v[t+1] - v[t]) / h`, a trajectory the simulator produced has a median base residual of 1e-12 body weights
  (worst single frame 5e-3, at a hard impact where the forward solver stops at 10 iterations). The actuated rows are
  the applied PD torques, and inverting the PD law returns the held setpoint to 1e-6 rad.
- **`r_soft` is mostly a foot-height meter.** In a static stand the soft-contact force is zero with the soles at
  0 mm, one body weight at about 0.24 mm of penetration, 8.5 body weights at 1 mm and 17 at 2 mm. `r_cone` is zero
  for all of these and one once the feet are out of reach of the floor.
- **Straight-line resampling cannot be differentiated.** On 1.2 s of feasible simulated motion sent through 30 FPS
  frames, the 95th-percentile `r_cone` is 0.0004 with an interpolating quintic spline, 0.05 with the default
  smoothing (its own bias, since these frames carry no noise) and 0.41 with the benchmark's linear resampling.
- **The default smoothing is a guess.** It has only seen noise-free simulated frames, where less smoothing is
  better. Real retargeted clips carry noise; `sweep` is how the right amount gets chosen.

Consequence for the projection stage: through MuJoCo's contact model as it stands, the base residual has no gradient
while a foot is above the floor and a slope of four to twelve body weights per millimetre once it is below. The
projection needs a relaxed contact model that is tightened as it converges, or explicit contact-force variables.

## Not yet run

- Anything on real LAFAN1 clips: the machine this was written on cannot reach Hugging Face.
- `dump_tracking.py` (needs JAX and a GPU).
- `test_model_matches_benchmark` (skipped without JAX).
