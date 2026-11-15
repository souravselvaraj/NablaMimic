# Aasaan

Feedback-distilled trajectory optimization for humanoid motion tracking, built next to ∇Mimic so every number is
measured on the benchmark's own G1, timestep and contact model. Work in progress; nothing in `src/nablamimic` is
changed.

## Experiment 0: does reference infeasibility predict tracking difficulty?

No training and no GPU. For every 200 Hz frame of every clip it asks how much force would have to act on the robot's
unactuated base for the reference to be a real motion, in units of body weight:

| residual | contact forces | what a large value means |
|---|---|---|
| `r_soft` | MuJoCo's soft contact, evaluated on the reference as given | mostly that the feet are a fraction of a millimetre off |
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

## Stages 1 and 2: the teacher

```bash
python -m aasaan.teach --clips <clip names> --start 5 --seconds 8 --out runs/teacher     # CPU; gates 1 and 2
```

**The plan changed here, on evidence.** Stage 1 was going to be a positions-only projection: optimize the motion
itself until MuJoCo's inverse dynamics leaves no force on the base. Through the benchmark's contact model that
problem cannot be descended: the base residual has no gradient while a foot is above the floor, a slope of four to
twelve body weights per millimetre once it is below, and friction forces that react to millimetres per second of
slip. A sparse Levenberg–Marquardt projection (colored finite-difference Jacobian, direct solves), started from a
reference whose feet had already been seated to carry the right vertical load, stalled at a median residual of
0.6 body weights.

What works instead (`teacher.py`):

1. **Feedforward from inverse dynamics with free contact forces.** Joint torques are read off inverse dynamics with
   the ground allowed any force inside the friction pyramids near the floor, choosing the set with the smallest
   joint torques. Inverting the PD law gives one setpoint per 20 ms control period.
2. **Iterative LQR in the simulator.** Linearize the 50 Hz map with MuJoCo's finite differences, run Riccati
   backward, roll forward in the real simulator, repeat along the rollout. The nominal that comes out is a rollout,
   so it is feasible by construction, and the same pass yields the gains `K` and the cost-to-go `P`.

Measured (`tests/test_aasaan_teacher.py`, CPU):

| reference | gate 1: one-period defect | open loop | teacher, closed loop |
|---|---|---|---|
| 0.8 s the simulator produced | root 0.007 mm, joints 1 mrad | 0.17 mm off after 0.4 s | not run |
| 2.4 s kinematic squat, never simulated | root 0.6 mm, joints 20 mrad (the ankles seating soles that are flat only to 1 mm) | falls at 2.2 s | stays within 1.0 mm and 1.8 mrad at the root |

Pushes on the squat teacher (impulse on the torso over 0.1 s):

| impulse | forward | backward | sideways | peak cost-to-go |
|---|---|---|---|---|
| 5 N s | recovers | recovers | recovers | under 400 |
| 10 N s | stays up, 116 mm off at the end | falls | recovers | 320 to 1e5 |
| 20 N s, 40 N s | falls | falls | falls | above 2.7e4 |

So linear feedback around a planted-feet reference absorbs roughly 5 to 10 N s. That is far below the 40 N s used
to test learned trackers, which recover by stepping; nothing here can choose a new foothold. Whether walking and
running references give wider tubes is the first thing to check on real clips. The cost-to-go separates the
recovered pushes from the falls by two orders of magnitude, which is what the value-triggered stage needs.

Cost: about 2.3 CPU-seconds per second of motion per backward pass, three passes by default.

The LQR weights in `teacher.TOL` are provisional tolerances, not yet the quadratic expansion of the benchmark reward.

## Not yet run

- Anything on real LAFAN1 clips: the machine this was written on cannot reach Hugging Face. Both the smoothing
  default and every teacher number above are from simulated or synthetic motion.
- `dump_tracking.py` (needs JAX and a GPU).
- `test_model_matches_benchmark` was skipped here for lack of JAX.
- The observation-only (LQG) teacher, distillation into the policy, and value-triggered correction do not exist yet.
