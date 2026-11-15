"""Experiment 0 on CPU MuJoCo. Needs Menagerie's G1 (scripts/setup_menagerie.sh); no GPU, no JAX, no LAFAN1.

The ground truth throughout is a trajectory the simulator itself produced (a G1 sagging and falling under a swaying
PD setpoint): it is feasible by construction, so both residuals must vanish on it.
"""

import csv

import mujoco
import numpy as np
import pytest

from aasaan import exp0
from aasaan.feasibility import Analyzer, summarize, windows
from aasaan.model import NBASE, SUBSTEPS, build_model, implied_setpoint, scene_path, torque_limits
from aasaan.reference import Reference, build_reference, kinematics, to_csv_rows

pytestmark = pytest.mark.skipif(not scene_path().exists(), reason="Menagerie G1 missing (scripts/setup_menagerie.sh)")


@pytest.fixture(scope="module")
def sim():
    """3 s of simulation at 200 Hz: qpos, qvel, held setpoints and actuator forces."""
    m, kp, kd = build_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("knees_bent").id)
    q0 = d.qpos[7:].copy()
    Q, V, U, F = [], [], [], []
    for k in range(150):
        d.ctrl[:] = q0 + 0.05 * np.sin(2 * np.pi * 0.7 * k * 0.02 + np.arange(m.nu))
        for _ in range(SUBSTEPS):
            mujoco.mj_forward(m, d)
            Q.append(d.qpos.copy()), V.append(d.qvel.copy()), U.append(d.ctrl.copy())
            F.append(d.qfrc_actuator.copy())
            mujoco.mj_step(m, d)
    Q.append(d.qpos.copy()), V.append(d.qvel.copy())
    return dict(m=m, kp=kp, kd=kd, Q=np.array(Q), V=np.array(V), U=np.array(U), F=np.array(F))


@pytest.fixture(scope="module")
def analyzer():
    return Analyzer()


def _sim_reference(sim):
    h = sim["m"].opt.timestep
    qvel, qacc = kinematics(sim["m"], sim["Q"], h)
    return Reference("sim", h, sim["Q"], qvel, qacc, 0.0, {})


def _resample_30fps(Q, h):
    """30 FPS frames of a 200 Hz trajectory by interpolation between neighbouring samples (wxyz qpos)."""
    t = np.arange(0.0, (len(Q) - 1) * h, 1.0 / 30.0)
    x = t / h
    i = np.minimum(np.floor(x).astype(int), len(Q) - 2)
    w = (x - i)[:, None]
    out = (1 - w) * Q[i] + w * Q[i + 1]
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def _soles_on_floor(m, key):
    """The keyframe pose shifted in z so its lowest floor distance is exactly zero."""
    geo, _, _ = build_model()
    floor = geo.geom("floor").id
    for p in range(geo.npair):
        if floor in (geo.pair_geom1[p], geo.pair_geom2[p]):
            geo.pair_margin[p] = 2.0
    d = mujoco.MjData(geo)
    q = m.key(key).qpos.copy()
    d.qpos[:] = q
    mujoco.mj_kinematics(geo, d)
    mujoco.mj_collision(geo, d)
    q[2] -= min(d.contact[i].dist for i in range(d.ncon) if floor in (d.contact[i].geom1, d.contact[i].geom2))
    return q


def test_model_matches_benchmark():
    jax = pytest.importorskip("jax")  # noqa: F841  nablamimic.env needs it
    from nablamimic.env import EnvConfig, build_model as bench_build
    mb, kpb = bench_build(EnvConfig())
    m, kp, _ = build_model()
    assert np.array_equal(kp, kpb)
    for name in ("actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "body_mass", "dof_armature",
                 "pair_friction", "pair_solref", "jnt_range"):
        assert np.array_equal(getattr(m, name), getattr(mb, name)), name
    for name in ("timestep", "iterations", "integrator"):
        assert getattr(m.opt, name) == getattr(mb.opt, name), name


def test_positions_alone_recover_velocity(sim):
    """v[t] = (q[t] - q[t-1]) / h is the simulator's own velocity, so positions are a complete description."""
    qvel, _ = kinematics(sim["m"], sim["Q"], sim["m"].opt.timestep)
    assert np.abs(qvel[1:] - sim["V"][1:]).max() < 1e-9


def test_simulated_motion_has_no_residual(sim, analyzer):
    """The floor of both diagnostics: a physically produced motion needs no force on its base."""
    diag = analyzer.analyze(_sim_reference(sim))
    assert np.percentile(diag.r_soft, 95) < 1e-6
    assert np.percentile(diag.r_cone, 95) < 1e-6
    # single frames at hard impacts keep the forward solver's truncation at 10 iterations
    assert diag.r_soft.max() < 0.02 and diag.r_cone.max() < 0.02
    assert not diag.flight.any()


def test_inverse_dynamics_recovers_actuation(sim):
    """Actuated rows of the discrete inverse are the PD torques the simulator applied, and inverting the PD law gives
    back the setpoint that was held (wherever the torque limit did not clip)."""
    mi, kp, kd = build_model(discrete_inverse=True)
    d = mujoco.MjData(mi)
    h, Q, V = mi.opt.timestep, sim["Q"], sim["V"]
    lim = torque_limits(mi)
    worst_tau, worst_u, checked = 0.0, 0.0, 0
    for t in range(1, len(Q) - 1):
        d.qpos[:], d.qvel[:], d.qacc[:] = Q[t], V[t], (V[t + 1] - V[t]) / h
        mujoco.mj_inverse(mi, d)
        tau = d.qfrc_inverse[NBASE:]
        free = np.abs(sim["F"][t][NBASE:]) < 0.99 * lim
        if np.abs(d.qfrc_inverse[:NBASE]).max() > 1e-6:  # an unconverged impact frame of the forward solver
            continue
        worst_tau = max(worst_tau, np.abs(tau - sim["F"][t][NBASE:]).max())
        u = implied_setpoint(tau, Q[t][7:], V[t][NBASE:], kp, kd)
        worst_u = max(worst_u, np.abs(u - sim["U"][t])[free].max(initial=0.0))
        checked += 1
    assert checked > 500
    assert worst_tau < 1e-6 and worst_u < 1e-6


@pytest.mark.parametrize("key", ["home", "knees_bent"])
def test_soft_residual_is_about_height_cone_residual_is_not(analyzer, key):
    """Why two residuals: a static stand is feasible, yet the soft-contact residual is a full body weight with the
    soles 5 mm up and many body weights with them 2 mm down. The cone residual is zero in both and one in the air."""
    q = _soles_on_floor(analyzer.m, key)
    zero = np.zeros(analyzer.m.nv)

    def at(dz):
        qq = q.copy()
        qq[2] += dz
        return analyzer.frame(qq, zero, zero)

    up, down, air = at(0.005), at(-0.002), at(1.0)
    assert up[0] == pytest.approx(1.0, abs=0.01) and up[1] < 1e-3      # hovering 5 mm: soft 1, cone 0
    assert down[0] > 5.0 and down[1] < 1e-3                            # 2 mm deep: soft explodes, cone 0
    assert air[1] == pytest.approx(1.0, abs=0.01) and air[5] == 0      # 1 m up: nothing can hold it
    assert up[7] < 0.5                                                 # standing needs well under the torque limits


def test_smoothing_keeps_a_feasible_clip_feasible(sim, analyzer):
    """Through the real path (30 FPS frames -> spline -> 200 Hz -> residuals) a feasible motion stays near zero,
    while the benchmark's straight-line resampling of the same frames does not: its acceleration lives in spikes at
    the source frames."""
    h = sim["m"].opt.timestep
    raw = _resample_30fps(sim["Q"][:241], h)  # the first 1.2 s, before the fall's hard impacts
    spline = summarize(analyzer.analyze(build_reference(analyzer.m, raw, "s", floor_align=False)), h)
    linear = summarize(analyzer.analyze(build_reference(analyzer.m, raw, "l", kind="linear", floor_align=False)), h)
    assert spline["r_cone_p50"] < 0.02 and spline["r_cone_p95"] < 0.15
    assert linear["r_cone_p95"] > 3 * spline["r_cone_p95"]
    assert linear["acc_p99"] > 2 * spline["acc_p99"]


def test_windows_cover_the_clip_in_control_frames(sim, analyzer):
    diag = analyzer.analyze(_sim_reference(sim))
    rows = windows(diag, sim["m"].opt.timestep)
    assert rows[0]["frame0"] == 0 and rows[-1]["frame1"] == 150
    assert all(a["frame1"] == b["frame0"] for a, b in zip(rows, rows[1:]))


def test_cli_run_and_join(sim, tmp_path, capsys):
    """run on a directory of CSVs in the dataset's layout, then join against a tracking-error table."""
    h = sim["m"].opt.timestep
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    raw = _resample_30fps(sim["Q"], h)
    for name, rows in (("clip_a", raw), ("clip_b", raw[:60])):
        np.savetxt(data / f"{name}.csv", to_csv_rows(rows), delimiter=",")
    exp0.main(["run", "--data", str(data), "--out", str(out), "--workers", "1"])
    with open(out / "summary.csv") as f:
        summary = list(csv.DictReader(f))
    assert [r["clip"] for r in summary] == ["clip_a", "clip_b"]
    assert (out / "frames" / "clip_a.npz").exists()
    with open(out / "windows.csv") as f:
        wins = list(csv.DictReader(f))
    # a made-up tracking error that follows the residual, to exercise the join end to end
    track = tmp_path / "tracking.csv"
    with open(track, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "frame", "error"])
        for r in wins:
            for fr in range(int(r["frame0"]), int(r["frame1"])):
                w.writerow([r["clip"], fr, 0.01 + float(r["r_cone_max"])])
    capsys.readouterr()
    exp0.main(["join", "--exp0", str(out), "--tracking", str(track)])
    text = capsys.readouterr().out
    assert "rank correlation, all windows:      +1.000" in text
