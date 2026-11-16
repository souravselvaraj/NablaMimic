# Aasaan

Feedback-distilled trajectory optimization for humanoid motion tracking, built next to ∇Mimic so every number is
measured on the benchmark's own G1, timestep and contact model. Work in progress; nothing in `src/nablamimic` is
changed.

**Status: the teacher does not work on real clips.** It holds every synthetic test in this file and produced no
feasible trajectory on any of the five LAFAN1 clips it was run on ("Real clips" below). What is left to find out
is which half failed, and `aasaan.hold` is the experiment for that.

## Experiment 0: does reference infeasibility predict tracking difficulty?

No training and no GPU. For every 200 Hz frame of every clip it asks how much force would have to act on the robot's
unactuated base for the reference to be a real motion, in units of body weight:

| residual | contact forces | what a large value means |
|---|---|---|
| `r_soft` | MuJoCo's soft contact, evaluated on the reference as given | mostly that the feet are a fraction of a millimetre off |
| `r_cone` | the best forces any friction pyramid allows at points within 2 cm of the floor | the motion itself breaks Newton–Euler |

```bash
python -m aasaan.exp0 sweep --clips <a few clip names> --refs runs/refs    # is the result stable as the smoothing changes?
python -m aasaan.exp0 run --refs runs/refs --out runs/exp0                 # all 40 clips, parallel over clips
python -m aasaan.dump_tracking --run runs/base_s0 --out runs/exp0/tracking.csv   # GPU; after the baseline exists
python -m aasaan.exp0 join --exp0 runs/exp0 --tracking runs/exp0/tracking.csv
```

Cost: the spline fit dominates and grows faster than the clip length. At 6,600 source frames one channel takes 75
to 120 s and a clip has 36, so a full clip is about an hour of one core; the residuals after it take seconds.
`--refs` keeps each fitted reference on disk, so a clip is fitted once per smoothing setting. (An earlier version
of this file said twelve minutes for the whole dataset. That was extrapolated from 1 s clips and was wrong.)

Read the `sweep` table before anything else. Where `r_cone` halves when the smoothing doubles, it is measuring the
differentiation of noisy 30 FPS frames, not the motion; trust only the range where it is flat.

`join` ranks windows by the mean `r_cone` in each. The window maximum was the first choice and is the wrong one: on
the real clips a single frame can carry it, and it moved by up to seven times with the smoothing (below).

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
  frames, the 95th-percentile `r_cone` is under 0.0001 with an interpolating quintic spline, 0.07 with the default
  smoothing (its own bias, since these frames carry no noise) and 0.36 with the benchmark's linear resampling.

### First numbers on real clips

Sweep over the smoothing on walk1_subject1, run1_subject2, jumps1_subject1, dance1_subject1 and
fallAndGetUp3_subject1 (run on the cluster, not reproduced here; full table in `runs/exp0/REPORT.md` there):

- **The residual measures the motion, not the noise.** From exact interpolation up to the default smoothing, the
  median and 95th percentile of `r_cone` move by under 20% on every clip. Doubling the smoothing from there raises
  the 95th percentile on walk (+39%) and run (+28%), and four times the default more than doubles it on four
  clips of five.
- **The default smoothing (2 mm, 0.002, 3 mrad) stays.** It is the top of that flat range and it removes
  single-frame spikes that interpolation keeps (largest `r_cone`: jumps 14.8 to 2.0, fall 9.7 to 1.9).
- **Walking is close to feasible, the dynamic clips are not.** 95th percentile `r_cone` at the default: walk 0.09,
  fall 0.30, dance 0.31, run 0.54, jumps 0.63 body weights.
- **Linear resampling is worse everywhere**: 1.9 to 3.7 times the 95th percentile, 6 to 31 times the maximum.
- **The simulator's contact carries nothing in the typical frame.** The median `r_soft` is 1.0 body weights on
  all five clips, which is what a robot with no floor under it gives. The likely reason, not yet checked frame by
  frame: the benchmark puts the floor at the 5th percentile of the lowest foot point, so the soles are usually
  just above it. This is what the seating step below exists for.

### Smoothing: what was tried

The fit is scipy's `make_splrep` (quintic, knots added until the RMS distance from the frames reaches eps). Two
faster fits were measured against it on simulated motion with a known answer and not adopted:

| fit | time, 6,600 frames, one channel | 95th percentile `r_cone`, 1.2 s feasible clip, default eps |
|---|---|---|
| `make_splrep` (in use) | 75 to 120 s | 0.07 |
| FITPACK `splrep`, same family | about 1 s | 0.07, but up to 45% apart on other clips and settings |
| cubic of least curvature at RMS distance eps | about 1 s | 0.20: it spends its budget flattening the sharpest accelerations |

The sweep above was run with `make_splrep`, so changing the fit means repeating it. With `--refs` the cost is paid
once per clip.

## Stages 1 and 2: the teacher

```bash
python -m aasaan.teach --clips <clip names> --start 5 --seconds 8 --refs runs/refs --out runs/teacher   # CPU; gates 1 and 2
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
2. **Seating.** Each reference state is moved vertically until the simulator's own contact carries the vertical
   load the motion needs there. Without it nothing below works on a clip whose feet are 1 mm above the floor: the
   simulator sees no contact, the linearization describes free fall, and the gains throw the robot over.
3. **Iterative LQR in the simulator.** Linearize the 50 Hz map with MuJoCo's finite differences along the seated
   states, run Riccati backward, roll forward in the real simulator, repeat along the rollout. If that first
   rollout falls, a second kind of first pass (bootstrap) linearizes at the state the simulator is actually in at
   every step. The nominal that comes out is a rollout, so it is feasible by construction, and the same pass yields
   the gains `K` and the cost-to-go `P`.

Measured (`tests/test_aasaan_teacher.py`, CPU). The squat is 2.4 s, kinematic, never simulated; its soles are flat
only to 1 mm.

| reference | seating moved it | gate 1, one period | open loop | first pass that held | teacher vs reference, root |
|---|---|---|---|---|---|
| 0.8 s the simulator produced | not needed | 0.007 mm, joints 1 mrad | 0.17 mm off after 0.4 s | not run | not run |
| squat as built | 0.2 mm | 0.5 mm, joints 20 mrad | falls at 2.2 s | seated | 1.0 mm |
| squat raised 3 mm | 3.2 mm | 0.5 mm | falls at 2.2 s | seated | 4.0 mm (3 of it the floor) |
| squat raised 18 mm, sunk 8 mm | 18 mm, 8 mm | 0.5 mm | | seated | 18.8 mm, 7.8 mm |
| squat with root and ankles drifting ±3 mm, ±10 mrad | up to 6.8 mm | 0.8 mm | falls at 2.3 s | bootstrap (seated fell) | 6.2 mm |

Before seating, the raised-by-1-mm squat produced no trajectory at all. The bootstrap alone is not enough either:
across six offsets, four weightings and two gain horizons it fell in 10 of 48 runs, with no pattern in which.

Pushes (impulse on the torso over 0.1 s), the same for the as-built, raised and drifting squats:

| impulse | forward | backward | sideways | peak cost-to-go |
|---|---|---|---|---|
| 5 N s | recovers | recovers | recovers | 76 to 400 |
| 10 N s | stays up, 2 to 12 cm off at the end | falls | recovers | 320 to 1e5 |
| 20 N s | falls | falls | falls | above 2.5e4 |

So linear feedback around a planted-feet reference absorbs roughly 5 to 10 N s; 10 N s changes this 33 kg robot's
velocity by 0.3 m/s. Recovering from more takes a step, and nothing here can choose a new foothold. Whether walking
and running references give wider tubes is the first thing to check on real clips. The cost-to-go separates the
recovered pushes from the falls by two orders of magnitude, which is what the value-triggered stage needs.

Cost: about 2.3 CPU-seconds per second of motion per linearization pass; three passes by default, four when the
bootstrap is needed.

Known weak points, none of them tested on real motion: seating only moves states vertically, so a foot that
slides in the reference is linearized as sliding; a tilted foot is seated on its edge; and the LQR weights in
`teacher.TOL` are provisional tolerances (2 cm on the root, where the benchmark reward's scale is 30 cm), not yet
the quadratic expansion of the benchmark reward.

## Real clips

Run on the cluster with commit `edf16ba`, default smoothing; numbers as reported from there, not reproduced here.

### The baseline this has to beat

∇Mimic as shipped, one RTX PRO 6000, seed 0: return 0.811 at 403 s of training and 0.832 at 503 s, so 0.82 is
crossed at about 7.5 minutes. Final return 0.908 after 3 h, no clip failing. Its weakest reward term is root
position, 0.66 on average: it reaches that return while letting the root wander (a constant error of about 20 cm
would give 0.66 at the reward's 30 cm scale).

### Teacher gates: all five fail

8 s windows starting at 5 s.

| clip | `r_cone` p95 | seating, median / max | gate 1 root | gate 1 joints | open loop | teacher |
|---|---|---|---|---|---|---|
| walk1_subject1 | 0.03 | 6.1 / 11.7 mm | 1.5 mm | 44 mrad | 1.1 s | fell at 1.3 s |
| run1_subject2 | 0.09 | 4.1 / 20.3 mm | 1.6 mm | 65 mrad | 1.5 s | fell at 1.3 s |
| jumps1_subject1 | 0.71 | 3.0 / 29.7 mm | 2.1 mm | 40 mrad | 2.0 s | fell at 1.6 s |
| dance1_subject1 | 0.13 | 8.3 / 18.6 mm | 1.5 mm | 31 mrad | 1.0 s | fell at 1.8 s |
| fallAndGetUp3_subject1 | 0.18 | 9.0 / 30.0 mm | 1.7 mm | 36 mrad | 0.7 s | fell at 0.7 s |

Feedback bought nothing over the feedforward alone. The walk window is the telling row: by `r_cone` it is almost
feasible (0.03 body weights) and it still cannot be held for a second and a half. So `r_cone` does not measure what the
teacher needs. It asks whether some forces at points within 2 cm of the floor could support the motion; the
simulator only supplies force at a sole that is on the floor, flat and not sliding. The references are far from
that (feet 3 to 9 mm off the floor in the typical frame, up to 30 mm), and the synthetic squat that every test here
uses has none of it: planted flat feet, no steps. The tests passed and said nothing about real motion.

### Experiment 0 at clip level: no better than acceleration

Over the 40 clips, against the baseline's per-clip return (Spearman):

| predictor | rank correlation |
|---|---|
| `r_cone` p95 | -0.68 |
| share of frames with `r_cone` above 0.1 | -0.68 |
| share of frames with nothing near the floor | -0.60 |
| joint acceleration p99 | -0.62 |
| `r_soft` p95 | +0.06 |

With 40 clips, -0.68 and -0.62 are not distinguishable. And the top of `r_cone` is not dynamics: 98% of the worst
1% of frames have nothing within 2 cm of the floor while the motion needs support (median 0.72 body weights), one
clip for 20 s on end. The simulator's contact carries nothing in 54 to 92% of frames (median clip 89%). The largest
infeasibility in these references is where the floor is, which the benchmark's reward is built to forgive.

### What decides the next step: `aasaan.hold`

```bash
python -m aasaan.hold --traj runs/hold/<clip>.npz ... --out runs/hold
```

The gates failed with a reference as the nominal. `hold` takes the nominal from a rollout a working tracker already
produced (states and PD targets at 50 Hz), so feasibility is given and only the feedback is tested: linearize along
the log, Riccati, roll out, push. It first checks that the file agrees with itself and that this simulator
reproduces it, because a wrong quaternion order or velocity frame would look exactly like feedback failing.

- If feedback holds walking and running logs: the feedback half is sound and the missing piece is how to get a
  first feasible trajectory from a reference. `teach` now reports how the reference treats the points it leans on
  (`hover_mm`, `slip_cm_s`, the joint that carries the gate 1 defect) to size that gap.
- If it does not: Riccati feedback at 50 Hz through this contact model cannot hold stepping motion, and the
  teacher is the wrong tool whatever the reference looks like.

On logs this simulator made itself (a sway, the squat teacher's rollout, a robot toppling to the floor) it
reproduces them to 1e-11 mm or better in the typical period and holds them; a float32 copy to 2e-5 mm. None of
those is a stepping motion.

## Not yet run

- `aasaan.hold` on any log from the baseline; the script that writes those logs does not exist yet.
- `dump_tracking.py` (needs JAX and a GPU) and `join`.
- Nothing in this directory has been run on real motion from this machine, which cannot reach the dataset.
- `test_model_matches_benchmark` was skipped here for lack of JAX.
- The observation-only (LQG) teacher, distillation into the policy, and value-triggered correction do not exist yet.
