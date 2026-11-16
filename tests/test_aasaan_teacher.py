"""Stages 1 and 2 on CPU MuJoCo: feedforward from inverse dynamics, then the iLQR teacher. Needs Menagerie's G1.

Two references with known character: a trajectory the simulator produced (feasible), and a kinematic squat that was
never simulated (plausible, not feasible, soles flat only to a millimetre).
"""

import csv
import json

import mujoco
import numpy as np
import pytest

from aasaan import hold, teach
from aasaan.feasibility import Analyzer, stance_quality
from aasaan.model import SUBSTEPS, build_model, scene_path
from aasaan.reference import Reference, kinematics, load_reference, to_csv_rows
from aasaan.synthetic import squat
from aasaan.teacher import (ControlPath, Setpoints, Simulator, control_states, feedforward, heading_pushes, ilqr,
                            one_period_defect, open_loop, rollout, seat)

pytestmark = pytest.mark.skipif(not scene_path().exists(), reason="Menagerie G1 missing (scripts/setup_menagerie.sh)")


@pytest.fixture(scope="module")
def an():
    return Analyzer()


@pytest.fixture(scope="module")
def sim():
    return Simulator()


@pytest.fixture(scope="module")
def simulated():
    """0.8 s the simulator produced itself: a G1 sagging under a swaying PD setpoint."""
    m, _, _ = build_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("knees_bent").id)
    q0 = d.qpos[7:].copy()
    Q = []
    for k in range(40):
        d.ctrl[:] = q0 + 0.05 * np.sin(2 * np.pi * 0.7 * k * 0.02 + np.arange(m.nu))
        for _ in range(SUBSTEPS):
            Q.append(d.qpos.copy())
            mujoco.mj_step(m, d)
    Q = np.array(Q + [d.qpos.copy()])
    h = m.opt.timestep
    qvel, qacc = kinematics(m, Q, h)
    return Reference("sim", h, Q, qvel, qacc, 0.0, {})


@pytest.fixture(scope="module")
def squat_case(an, sim):
    ref = squat(an.m, seconds=2.4)
    ff = feedforward(an, ref)
    return ref, ff, ilqr(sim, ref, ff, passes=1)


def test_feedforward_replays_a_simulated_motion(an, sim, simulated):
    """Gate 1 where the answer is known: setpoints recovered from inverse dynamics, with contact forces chosen by
    the pyramid solve rather than read from the simulator, reproduce the simulator's own motion."""
    ff = feedforward(an, simulated)
    assert np.percentile(ff.r_cone, 95) < 0.01
    de = one_period_defect(sim, simulated, ff)
    assert np.median(de[:, 0]) < 5e-5 and de[:, 0].max() < 5e-4      # root: 0.05 mm typical, 0.5 mm worst
    assert np.median(de[:, 2]) < 5e-3                                # joints: 5 mrad
    ol = rollout(sim, simulated, open_loop(simulated, ff), ff=ff)
    assert ol.fell < 0 and ol.err[20, 0] < 2e-3                      # 0.4 s open loop: within 2 mm


def test_linearization_predicts_the_simulator(an, sim):
    """A and B of the 50 Hz map against the map itself, at a stance state of the squat."""
    ref = squat(an.m, seconds=1.0)
    ff = feedforward(an, ref)
    Q, V = control_states(ref)
    k = 30
    A, B = sim.linearize(Q[k], V[k], ff.u[k])
    q1, v1 = sim.step(Q[k], V[k], ff.u[k])
    rng = np.random.default_rng(0)
    dx = rng.normal(size=2 * sim.nv) * 1e-4
    du = rng.normal(size=sim.nu) * 1e-3
    q = Q[k].copy()
    mujoco.mj_integratePos(sim.m, q, dx[:sim.nv], 1.0)
    q2, v2 = sim.step(q, V[k] + dx[sim.nv:], ff.u[k] + du)
    actual = sim.error(q1, v1, q2, v2)
    predicted = A @ dx + B @ du
    assert np.linalg.norm(actual - predicted) < 0.1 * np.linalg.norm(actual)


def test_teacher_turns_a_kinematic_squat_into_a_trajectory(an, sim, squat_case):
    """The feedforward alone drifts away; the iLQR nominal is a real rollout that stays on the reference, and the
    second pass lowered the tracking cost."""
    ref, ff, teacher = squat_case
    ol = rollout(sim, ref, open_loop(ref, ff), ff=ff)
    cl = rollout(sim, ref, teacher, ff=ff)
    assert teacher.feasible and cl.fell < 0
    assert len(teacher.cost) == 2 and teacher.cost[1] < teacher.cost[0]
    assert cl.err[:, 0].max() < 3e-3 and cl.err[:, 1].max() < 0.01   # root within 3 mm and 10 mrad of the reference
    assert ol.fell >= 0 or ol.err[:, 0].max() > 10 * cl.err[:, 0].max()
    assert cl.value.max() < 1e-6                                     # undisturbed, it replays its own nominal


def test_teacher_recovers_from_a_small_push_and_its_value_sees_a_large_one(sim, squat_case):
    """5 N s on the torso: back on the reference by the end, cost-to-go decayed. 40 N s is far outside what linear
    feedback around planted feet can absorb; the teacher's own value says so by orders of magnitude."""
    ref, ff, teacher = squat_case
    k = teacher.k0 + 30
    peaks = {}
    for name, imp, dirv in (("small_fwd", 5.0, (1, 0, 0)), ("small_side", 5.0, (0, 1, 0)), ("large", 40.0, (1, 0, 0))):
        push = {j: np.array(dirv, float) * imp / 0.1 for j in range(k, k + 5)}
        r = rollout(sim, ref, teacher, push=push, ff=ff)
        peaks[name] = r.value.max()
        if imp == 5.0:
            assert r.fell < 0
            assert r.err[:, 0].max() > 5e-3 and r.err[-1, 0] < 5e-3      # knocked off by over 5 mm, back within 5 mm
            assert r.value[-1] < 0.05 * r.value.max()
    assert peaks["large"] > 20 * max(peaks["small_fwd"], peaks["small_side"])


def _moved(an, ref, name, dz, wobble=0.0, seed=0):
    """The reference with its root raised by dz and, with wobble, the root height and ankle angles drifting by
    that many metres / 3.3 x that many radians at a few hertz: feet that are off the floor the way retargeted
    clips' are."""
    Q = ref.qpos.copy()
    Q[:, 2] += dz
    if wobble:
        rng = np.random.default_rng(seed)
        t = np.arange(len(Q)) * ref.h

        def drift(amp):
            return amp * sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) for f in (2.1, 3.0, 4.8)) / 2

        Q[:, 2] += drift(wobble)
        for i in range(an.m.nu):
            if "ankle" in an.m.joint(an.m.actuator_trnid[i, 0]).name:
                Q[:, 7 + i] += drift(3.3 * wobble)
    qvel, qacc = kinematics(an.m, Q, ref.h)
    return Reference(name, ref.h, Q, qvel, qacc, 0.0, {})


def test_feet_a_few_millimetres_off_the_floor_need_seating(an, sim):
    """Raised by 3 mm the reference never touches the floor as far as the simulator can tell. Seating finds the
    3 mm, gate 1 from seated states measures the feedforward again, and the teacher holds the clip at the height
    the floor allows."""
    ref = _moved(an, squat(an.m, seconds=2.0), "raised", 0.003)
    ff = feedforward(an, ref)
    qpos, shift = seat(an, ref)
    assert np.all(shift[1:] < -0.0029) and np.all(shift[1:] > -0.0036)
    raw, seated = one_period_defect(sim, ref, ff), one_period_defect(sim, ref, ff, qpos)
    assert np.median(raw[:, 0]) > 1.5e-3 and np.median(seated[:, 0]) < 1e-3
    teacher = ilqr(sim, ref, ff, qpos, passes=0)
    cl = rollout(sim, ref, teacher, ff=ff)
    assert teacher.feasible and teacher.start == "seated" and cl.fell < 0
    assert cl.err[:, 0].max() < 6e-3                                 # 3 mm of it is the floor


def test_teacher_survives_feet_that_wobble_about_the_floor(an, sim):
    """Root height and ankles drifting by +-3 mm and +-10 mrad around a 2 mm hover. Whichever first pass it takes,
    a real trajectory comes out and stays near the reference."""
    ref = _moved(an, squat(an.m, seconds=2.0), "wobble", 0.002, wobble=0.003)
    ff = feedforward(an, ref)
    qpos, shift = seat(an, ref)
    assert np.ptp(shift[1:]) > 0.003
    teacher = ilqr(sim, ref, ff, qpos, passes=1)
    cl = rollout(sim, ref, teacher, ff=ff)
    assert teacher.feasible and teacher.start in ("seated", "bootstrap") and cl.fell < 0
    assert cl.err[:, 0].max() < 0.015 and cl.err[:, 1].max() < 0.05


def test_cli_reports_the_gates(an, tmp_path, capsys):
    """teach on a CSV in the dataset's layout: the squat sent through 30 FPS frames. The smoothed reference is kept
    on disk and comes back bit for bit."""
    ref = squat(an.m, seconds=2.5)
    t = np.arange(0.0, (len(ref.qpos) - 1) * ref.h, 1.0 / 30.0)
    raw = ref.qpos[np.round(t / ref.h).astype(int)]
    data, out, refs = tmp_path / "data", tmp_path / "out", tmp_path / "refs"
    data.mkdir()
    np.savetxt(data / "squat.csv", to_csv_rows(raw), delimiter=",")
    teach.main(["--data", str(data), "--out", str(out), "--refs", str(refs), "--start", "0.2", "--seconds", "1.6",
                "--passes", "1", "--impulses", "5", "--workers", "1"])
    text = capsys.readouterr().out
    assert "squat" in text and "pushes survived" in text and ("seated" in text or "bootstrap" in text)
    z = np.load(out / "squat.npz")
    assert z["K"].shape[1:] == (29, 70) and z["q"].shape[0] == z["u"].shape[0] + 1
    kept = list(refs.glob("squat__spline_*.npz"))
    assert len(kept) == 1
    first = load_reference(an.m, data / "squat.csv", cache=refs)
    fresh = load_reference(an.m, data / "squat.csv")
    assert np.array_equal(first.qpos, fresh.qpos) and np.array_equal(first.qacc, fresh.qacc)
    assert first.z_shift == fresh.z_shift and first.fit_rms == fresh.fit_rms



def _log(seconds, kick=0.0):
    """(qpos, qvel, ctrl) at 50 Hz of the simulator under a swaying setpoint; with kick, shoved so that it topples."""
    m, _, _ = build_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("knees_bent").id)
    q0 = d.qpos[7:].copy()
    d.qvel[0], d.qvel[5] = kick, 1.25 * kick
    Q, V, U = [d.qpos.copy()], [d.qvel.copy()], []
    for k in range(int(round(seconds / 0.02))):
        d.ctrl[:] = q0 + 0.05 * np.sin(2 * np.pi * 0.7 * k * 0.02 + np.arange(m.nu))
        U.append(d.ctrl.copy())
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        Q.append(d.qpos.copy()), V.append(d.qvel.copy())
    return np.array(Q), np.array(V), np.array(U)


def test_stance_quality_tells_planted_feet_from_sliding_ones(an):
    """The squat leans on points that are at the floor and at rest. Drag the same squat sideways at 10 cm/s and
    lift it 5 mm and the points it leans on hover and slide by that much."""
    ref = squat(an.m, seconds=1.0)
    planted = stance_quality(an, ref)
    assert planted["loaded_frac"] == 1.0 and abs(planted["hover_mm"]) < 1.5 and planted["slip_cm_s"] < 0.5
    Q = ref.qpos.copy()
    Q[:, 1] += 0.10 * np.arange(len(Q)) * ref.h
    Q[:, 2] += 0.005
    qvel, qacc = kinematics(an.m, Q, ref.h)
    dragged = stance_quality(an, Reference("dragged", ref.h, Q, qvel, qacc, 0.0, {}))
    assert dragged["slip_cm_s"] == pytest.approx(10.0, abs=0.5)
    assert dragged["hover_mm"] == pytest.approx(planted["hover_mm"] + 5.0, abs=0.5)


def test_hold_reads_a_log_and_catches_wrong_conventions(sim, tmp_path, capsys):
    """A log this simulator made is reproduced exactly and held, with pushes. The same log with the quaternion
    written x y z w, or with the angular velocity in the world frame, is reported as disagreeing with itself; with
    the targets off by one period, as not reproduced."""
    Q, V, U = _log(3.0, kick=0.4)  # long enough to end lying down: frames only differ once the body has turned
    path, ff = ControlPath("log", Q, V), Setpoints(U)
    assert max(k for k in hold.kinematic_check(sim, path) if k == k) < 0.1
    own = one_period_defect(sim, path, ff, k0=0)[:, 0]
    assert np.median(own) < 1e-12 and own.max() < 1e-6    # not bit for bit: each period restarts the contact solver
    xyzw = Q.copy()
    xyzw[:, 3:7] = xyzw[:, [4, 5, 6, 3]]
    assert hold.kinematic_check(sim, ControlPath("xyzw", xyzw, V))[1] > 0.5
    world, R = V.copy(), np.zeros(9)
    for k in range(len(Q)):
        mujoco.mju_quat2Mat(R, Q[k, 3:7])
        world[k, 3:6] = R.reshape(3, 3) @ V[k, 3:6]
    assert hold.kinematic_check(sim, ControlPath("world", Q, world))[1] > 0.2
    late = one_period_defect(sim, ControlPath("late", Q[:-1], V[:-1]), Setpoints(U[1:]), k0=0)
    assert np.median(late[:, 2]) > 2e-4

    files = []
    for name, (q, v, u) in (("sway", _log(1.0)), ("late", (Q[:-1], V[:-1], U[1:]))):
        files.append(str(tmp_path / f"{name}.npz"))
        np.savez(files[-1], qpos=q.astype(np.float32), qvel=v.astype(np.float32), ctrl=u.astype(np.float32))
    hold.main(["--traj", *files, "--out", str(tmp_path / "out"), "--impulses", "5", "--instants", "2",
               "--workers", "1"])
    text = capsys.readouterr().out
    assert "WARNING late" in text and "WARNING sway" not in text
    assert "recovered / still up / fell" in text and (tmp_path / "out" / "teachers" / "sway.npz").exists()
    with open(tmp_path / "out" / "hold.csv") as f:
        row = next(r for r in csv.DictReader(f) if r["clip"] == "sway")
    assert sum(int(c) for c in row["push_5"].split("/")) == 6 and row["push_5_detail"].count("@") == 6
    with open(tmp_path / "out" / "push_plan.json") as f:
        plan = json.load(f)
    assert plan["duration_steps"] == 5 and [m["step"] for m in plan["clips"]["sway"]] == [17, 33]
    assert np.allclose(plan["clips"]["sway"][0]["side"], heading_pushes(_log(1.0)[0][17])["side"], atol=1e-4)
    with pytest.raises(ValueError, match="expected qpos"):
        hold.load(_bad_shape(tmp_path, Q, V, U), sim)


def test_pushes_are_named_by_the_robots_heading():
    """Facing +y (a quarter turn to the left), forward is +y and the robot's left is -x."""
    q = np.zeros(36)
    q[3], q[6] = np.cos(np.pi / 4), np.sin(np.pi / 4)
    d = heading_pushes(q)
    assert np.allclose(d["fwd"], [0, 1, 0], atol=1e-12) and np.allclose(d["side"], [-1, 0, 0], atol=1e-12)
    assert np.allclose(d["back"], -d["fwd"])


def _bad_shape(tmp_path, Q, V, U):
    f = tmp_path / "bad.npz"
    np.savez(f, qpos=Q, qvel=V, ctrl=U[:-1])
    return f
