"""Stages 1 and 2 on CPU MuJoCo: feedforward from inverse dynamics, then the iLQR teacher. Needs Menagerie's G1.

Two references with known character: a trajectory the simulator produced (feasible), and a kinematic squat that was
never simulated (plausible, not feasible, soles flat only to a millimetre).
"""

import mujoco
import numpy as np
import pytest

from aasaan import teach
from aasaan.feasibility import Analyzer
from aasaan.model import SUBSTEPS, build_model, scene_path
from aasaan.reference import Reference, kinematics, to_csv_rows
from aasaan.synthetic import squat
from aasaan.teacher import Simulator, control_states, feedforward, ilqr, one_period_defect, open_loop, rollout

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
    return ref, ff, ilqr(sim, ref, ff, passes=2)


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
    """The feedforward alone drifts away; the iLQR nominal is a real rollout that stays on the reference, and each
    pass lowered the tracking cost."""
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


def test_cli_reports_the_gates(an, tmp_path, capsys):
    """teach on a CSV in the dataset's layout: the squat sent through 30 FPS frames."""
    ref = squat(an.m, seconds=2.5)
    t = np.arange(0.0, (len(ref.qpos) - 1) * ref.h, 1.0 / 30.0)
    raw = ref.qpos[np.round(t / ref.h).astype(int)]
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    np.savetxt(data / "squat.csv", to_csv_rows(raw), delimiter=",")
    teach.main(["--data", str(data), "--out", str(out), "--start", "0.2", "--seconds", "1.6", "--passes", "1",
                "--impulses", "5", "--workers", "1"])
    text = capsys.readouterr().out
    assert "squat" in text and "pushes survived" in text
    z = np.load(out / "squat.npz")
    assert z["K"].shape[1:] == (29, 70) and z["q"].shape[0] == z["u"].shape[0] + 1
