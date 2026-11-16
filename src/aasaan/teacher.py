"""Stages 1 and 2: from a smooth reference to a feasible trajectory and a feedback controller around it.

Stage 1, feedforward. For every physics sample the motion's joint torques are read off inverse dynamics, with the
ground free to push anywhere inside the friction pyramid at every point near the floor (feasibility.Analyzer). Among
the force sets that zero the base rows, the one with the smallest joint torques is taken. Inverting the PD law
turns the torques into joint setpoints, averaged over each 20 ms control period because the policy holds one setpoint
per period.

Why not optimize positions through MuJoCo's own contact model, as first planned: its inverse is too stiff to descend
(one body weight per 0.24 mm of sole height, friction forces that react to millimetres per second of slip). A
sparse Levenberg-Marquardt projection through it stalled at 0.6 body weights of residual; the feedforward above,
replayed in the simulator, follows the same reference to about a millimetre. See src/aasaan/README.md.

Stage 2, feedback. Iterative LQR in the benchmark simulator itself. The 50 Hz map x[k+1] = F(x[k], u[k]) (four
physics steps, PD actuators, contacts) is linearized with MuJoCo's finite differences, a backward Riccati pass gives
gains and the quadratic cost-to-go, and a forward pass plays them in the simulator.

The first pass cannot linearize along the reference as given. In retargeted clips the feet sit a millimetre above
or below the floor in most frames; one millimetre above is no contact at all to the simulator, so the linearization
describes a robot in free fall and its gains throw the robot over (measured on a squat raised by 1 mm). seat() moves
each reference state vertically until the simulator's contact carries the vertical load the motion needs, and the
first pass linearizes along those states. If that pass falls, bootstrap() tries a rollout whose gains come from
linearizing at the state the simulator is actually in. Later passes are ordinary iLQR along the previous rollout.

The result is a nominal trajectory that came out of the simulator (feasible by construction) with the teacher

    u = u_nom[k] - K[k] dx,      dx = (q - q_nom[k], v - v_nom[k]) in tangent coordinates

and V_k(dx) = dx' P[k] dx, the local cost-to-go that later stages use as critic and as recovery-region estimate.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import nnls

from .model import NBASE, SUBSTEPS, build_model, implied_setpoint

# Provisional LQR weights, as 1 / tolerance^2. To be replaced by the quadratic expansion of the benchmark reward.
TOL = dict(root_pos=0.02, root_rot=0.05, joint=0.1, root_lin=0.2, root_ang=0.5, joint_vel=2.0, setpoint=0.1)
TERM_HEIGHT, TERM_TILT = 0.25, 0.8  # the benchmark's termination test (nablamimic.env.EnvConfig)


@dataclass
class Feedforward:
    u: np.ndarray           # (K, nu) setpoint of control period k (physics samples 4k .. 4k + 3)
    tau: np.ndarray         # (T, nu) joint torques at the physics rate
    r_cone: np.ndarray      # (T,) base residual left after the best contact forces, in body weights
    tau_ratio: np.ndarray   # (T,) max |torque| / limit
    spread: np.ndarray      # (K,) largest deviation of a substep's implied setpoint from its period mean (rad)


def feedforward(an, ref, reg=1e-3):
    """Setpoints that make the simulator's PD actuators produce the torques the reference needs.

    an: feasibility.Analyzer. reg weighs joint torque (as a fraction of each limit) against base residual when
    choosing contact forces: small enough that the base rows still close, large enough to pick one force set."""
    m, T = an.m, len(ref.qpos)
    nu = m.nu
    tau, r_cone = np.zeros((T, nu)), np.zeros(T)
    s = np.sqrt(reg) / an.tau_max
    df = an.d_free
    for t in range(1, T - 1):
        df.qpos[:], df.qvel[:], df.qacc[:] = ref.qpos[t], ref.qvel[t], ref.qacc[t]
        mujoco.mj_inverse(an.m_free, df)
        need = df.qfrc_inverse.copy()
        C = an._candidates(ref.qpos[t])
        if C.shape[1]:
            lhs = np.vstack([C[:NBASE] * an.scale[:, None], C[NBASE:] * s[:, None]])
            rhs = np.concatenate([need[:NBASE] * an.scale, need[NBASE:] * s])
            lam, _ = nnls(lhs, rhs)
            need = need - C @ lam
        tau[t], r_cone[t] = need[NBASE:], an.norm(need[:NBASE])
    tau[0], tau[-1], r_cone[0], r_cone[-1] = tau[1], tau[-2], r_cone[1], r_cone[-2]
    u_fine = implied_setpoint(tau, ref.qpos[:, 7:], ref.qvel[:, NBASE:], an.kp, an.kd)
    u_fine[0] = u_fine[1]  # qvel[0] is undefined
    K = (T - 1) // SUBSTEPS
    per = u_fine[:K * SUBSTEPS].reshape(K, SUBSTEPS, nu)
    u = per.mean(1)
    spread = np.abs(per - u[:, None]).max((1, 2))
    u = np.clip(u, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
    return Feedforward(u, tau, r_cone, (np.abs(tau) / an.tau_max).max(1), spread)


class Simulator:
    """The benchmark's forward physics, one 20 ms control period at a time."""

    def __init__(self, solver_iterations=10):
        self.m, self.kp, self.kd = build_model(solver_iterations)
        self.d = mujoco.MjData(self.m)
        self.nv, self.nu = self.m.nv, self.m.nu
        self.lo, self.hi = self.m.actuator_ctrlrange[:, 0].copy(), self.m.actuator_ctrlrange[:, 1].copy()
        self.torso = self.m.body("torso_link").id

    def _set(self, q, v, u):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:], self.d.qvel[:] = q, v
        self.d.ctrl[:] = np.clip(u, self.lo, self.hi)

    def step(self, q, v, u, force=None):
        """State after one control period. force: world-frame force (N) on the torso for the whole period."""
        self._set(q, v, u)
        if force is not None:
            self.d.xfrc_applied[self.torso, :3] = force
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)
        return self.d.qpos.copy(), self.d.qvel.copy()

    def linearize(self, q, v, u, eps=1e-6):
        """(A, B) of the control-period map in tangent coordinates dx = (dq, dv): four physics-step Jacobians from
        mjd_transitionFD, chained with the setpoint held."""
        n = 2 * self.nv
        self._set(q, v, u)
        A, B = np.eye(n), np.zeros((n, self.nu))
        a, b = np.zeros((n, n)), np.zeros((n, self.nu))
        for _ in range(SUBSTEPS):
            mujoco.mjd_transitionFD(self.m, self.d, eps, True, a, b, None, None)
            A, B = a @ A, a @ B + b
            mujoco.mj_step(self.m, self.d)
        return A, B

    def error(self, q_ref, v_ref, q, v):
        """dx = (q - q_ref, v - v_ref) with the position difference in tangent coordinates."""
        dx = np.zeros(2 * self.nv)
        mujoco.mj_differentiatePos(self.m, dx[:self.nv], 1.0, q_ref, q)
        dx[self.nv:] = v - v_ref
        return dx


@dataclass
class ControlPath:
    """States already at the control rate, for example a logged rollout: state k + 1 follows state k after one
    20 ms period. Stands in for a Reference wherever only the control-step states are used."""
    name: str
    qpos: np.ndarray    # (K, nq), quaternion (w, x, y, z)
    qvel: np.ndarray    # (K, nv)


@dataclass
class Setpoints:
    """One setpoint per control step, where a Feedforward is expected but only its u is used."""
    u: np.ndarray       # (K - 1, nu)


def control_states(ref):
    """States at the control steps. For a Reference: physics samples 0, 4, 8, ... (sample 0 has no velocity, so
    everything built on a Reference starts at control step 1)."""
    if isinstance(ref, ControlPath):
        return ref.qpos, ref.qvel
    return ref.qpos[::SUBSTEPS], ref.qvel[::SUBSTEPS]


def _sizes(dx, nv):
    """Largest components of a state error: root m, root rad, joint rad, velocity."""
    return (np.abs(dx[:3]).max(), np.abs(dx[3:6]).max(), np.abs(dx[6:nv]).max(), np.abs(dx[nv:]).max())


def seat(an, ref, reach=0.03, iters=18):
    """Reference positions at the control steps, each moved vertically until the simulator's own contact model
    carries the vertical load the motion needs at that instant; where the motion needs none, lifted just clear of
    the floor. Joint angles, orientation and horizontal position are untouched. States with nothing within `reach`
    of the floor are left alone. Returns (qpos (K, nq), shift (K,) in metres, negative = lowered).

    an: feasibility.Analyzer (its discrete inverse dynamics gives the contact force of a state directly)."""
    Q, V = control_states(ref)
    A = ref.qacc[::SUBSTEPS]
    d, df = an.d, an.d_free
    out, shift = Q.copy(), np.zeros(len(Q))

    def support(q, k, dz):
        d.qpos[:], d.qvel[:], d.qacc[:] = q, V[k], A[k]
        d.qpos[2] += dz
        mujoco.mj_inverse(an.m, d)
        return d.qfrc_constraint[2]

    for k in range(1, len(Q)):
        df.qpos[:], df.qvel[:], df.qacc[:] = Q[k], V[k], A[k]
        mujoco.mj_inverse(an.m_free, df)
        need = df.qfrc_inverse[2]
        if need < 0.05 * an.mg:  # flight: only make sure nothing is pressed into the floor
            if support(Q[k], k, 0.0) <= 1e-3 * an.mg:
                continue
            need = 1e-3 * an.mg
        lo, hi = -reach, reach  # support falls from large to zero as the state is raised
        if support(Q[k], k, lo) < need:
            continue
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if support(Q[k], k, mid) > need else (lo, mid)
        shift[k] = 0.5 * (lo + hi)
        out[k, 2] += shift[k]
    return out, shift


def one_period_defect(sim, ref, ff, qpos=None, k0=1, joints=False):
    """Gate 1. From each reference state, apply that period's setpoint for one control period in the real simulator
    and measure how far the result is from the next reference state. Rows: root m, root rad, joint rad, velocity
    (largest components) for control steps k0 .. K - 2. qpos: seated positions (seat) to start from and compare
    with, so the gate measures the feedforward and not the height of the feet. joints=True also returns the
    position defect of every joint, (rows, nu)."""
    Q, V = control_states(ref)
    Q = Q if qpos is None else qpos
    rows, per = [], []
    for k in range(k0, min(len(ff.u), len(Q) - 1)):
        q1, v1 = sim.step(Q[k], V[k], ff.u[k])
        dx = sim.error(Q[k + 1], V[k + 1], q1, v1)
        rows.append(_sizes(dx, sim.nv)), per.append(np.abs(dx[6:sim.nv]))
    return (np.array(rows), np.array(per)) if joints else np.array(rows)


def weights(nv, nu, tol=None):
    """Diagonal state and setpoint weights 1 / tolerance^2 (TOL)."""
    t = {**TOL, **(tol or {})}
    qd = np.concatenate([[t["root_pos"]] * 3, [t["root_rot"]] * 3, [t["joint"]] * (nv - 6),
                         [t["root_lin"]] * 3, [t["root_ang"]] * 3, [t["joint_vel"]] * (nv - 6)])
    return np.diag(1.0 / qd ** 2), np.eye(nu) / t["setpoint"] ** 2


def _gravity_z(q):
    """z component of gravity's direction in the root frame (-1 upright), from the root quaternion (w, x, y, z)."""
    _, x, y, _ = q[3:7]
    return -(1.0 - 2.0 * (x * x + y * y))


def _terminated(q, q_ref):
    return abs(q[2] - q_ref[2]) > TERM_HEIGHT or abs(_gravity_z(q) - _gravity_z(q_ref)) > TERM_TILT


@dataclass
class Teacher:
    """A nominal trajectory over control steps k0 .. k0 + N and the feedback law around it."""
    q: np.ndarray       # (N + 1, nq)
    v: np.ndarray       # (N + 1, nv)
    u: np.ndarray       # (N, nu)
    K: np.ndarray       # (N, nu, 2 nv)
    P: np.ndarray       # (N + 1, 2 nv, 2 nv)
    k0: int
    feasible: bool      # True if q, v, u are a simulator rollout that reached the end of the window
    cost: list          # tracking cost of the nominal after the bootstrap and after each accepted pass
    fell: int = -1      # control step at which the bootstrap tripped the termination test (then q, v, u stop there)
    start: str = ""     # which first pass gave the first real trajectory: "seated" or "bootstrap"
    seated_fell: int = -1   # control step at which the seated first pass fell, if it did

    def action(self, sim, i, q, v):
        """The teacher's setpoint at step k0 + i in state (q, v), with the state error it acted on."""
        dx = sim.error(self.q[i], self.v[i], q, v)
        return self.u[i] - self.K[i] @ dx, dx

    def value(self, i, dx):
        return float(dx @ self.P[i] @ dx)


@dataclass
class Rollout:
    q: np.ndarray       # (n + 1, nq) simulated states
    v: np.ndarray
    u: np.ndarray       # (n, nu) setpoints applied (clipped)
    err: np.ndarray     # (n + 1, 4) error against the reference: root m, root rad, joint rad, velocity
    value: np.ndarray   # (n + 1,) the teacher's cost-to-go of the deviation from its nominal
    fell: int           # control step at which the benchmark's termination test tripped, or -1
    cost: float         # tracking cost against the reference (inf after a fall)


def rollout(sim, ref, teacher, push=None, ff=None, tol=None, step=None, alpha=1.0):
    """Play the teacher in the simulator from its first nominal state.

    push: {control step: world force in N on the torso for that period}. ff: the Feedforward whose setpoints the
    cost measures effort against (default: the teacher's own nominal). step: (N, nu) open-loop corrections of an
    iLQR forward pass, scaled by alpha. A teacher with K = 0 plays its nominal setpoints open loop."""
    Qr, Vr = control_states(ref)
    Qc, R = weights(sim.nv, sim.nu, tol)
    k0, N = teacher.k0, len(teacher.u)
    q, v = teacher.q[0].copy(), teacher.v[0].copy()
    qs, vs, us, errs, vals, fell, cost = [q], [v], [], [], [], -1, 0.0
    for i in range(N):
        k = k0 + i
        e = sim.error(Qr[k], Vr[k], q, v)
        u, dx = teacher.action(sim, i, q, v)
        if step is not None:
            u = u + alpha * step[i]
        u = np.clip(u, sim.lo, sim.hi)
        du = u - (ff.u[k] if ff is not None else teacher.u[i])
        cost += e @ Qc @ e + du @ R @ du
        errs.append(_sizes(e, sim.nv)), vals.append(teacher.value(i, dx)), us.append(u)
        q, v = sim.step(q, v, u, None if push is None else push.get(k))
        qs.append(q), vs.append(v)
        if _terminated(q, Qr[k + 1]):
            fell, cost = k + 1, np.inf
            break
    i = len(us)
    e = sim.error(Qr[k0 + i], Vr[k0 + i], q, v)
    errs.append(_sizes(e, sim.nv))
    vals.append(teacher.value(i, sim.error(teacher.q[i], teacher.v[i], q, v)))
    if fell < 0:
        cost += 10.0 * e @ Qc @ e
    return Rollout(np.array(qs), np.array(vs), np.array(us), np.array(errs), np.array(vals), fell, float(cost))


def open_loop(ref, ff, k0=1, k1=None, qpos=None):
    """The feedforward alone as a Teacher with zero gains, along the reference states or their seated positions
    (not a real trajectory either way)."""
    Q, V = control_states(ref)
    Q = Q if qpos is None else qpos
    k1 = min(len(ff.u), len(Q) - 1) if k1 is None else k1
    N, n = k1 - k0, 2 * V.shape[1]
    return Teacher(Q[k0:k1 + 1].copy(), V[k0:k1 + 1].copy(), ff.u[k0:k1].copy(), np.zeros((N, ff.u.shape[1], n)),
                   np.zeros((N + 1, n, n)), k0, False, [])


def _backward(sim, ref, ff, nom, tol, terminal=10.0):
    """One Riccati pass along the nominal. Returns (open-loop steps, K, P): the Gauss-Newton iLQR recursion for the
    cost  sum e'Qe + (u - u_ff)'R(u - u_ff) + terminal e_N'Qe_N  with e the error against the reference."""
    Qr, Vr = control_states(ref)
    Qc, R = weights(sim.nv, sim.nu, tol)
    k0, N, n = nom.k0, len(nom.u), 2 * sim.nv
    step, K, P = np.zeros((N, sim.nu)), np.zeros((N, sim.nu, n)), np.zeros((N + 1, n, n))
    P[N] = terminal * Qc
    p = P[N] @ sim.error(Qr[k0 + N], Vr[k0 + N], nom.q[N], nom.v[N])
    for i in range(N - 1, -1, -1):
        A, B = sim.linearize(nom.q[i], nom.v[i], nom.u[i])
        e = sim.error(Qr[k0 + i], Vr[k0 + i], nom.q[i], nom.v[i])
        qx, qu = Qc @ e + A.T @ p, R @ (nom.u[i] - ff.u[k0 + i]) + B.T @ p
        BtP = B.T @ P[i + 1]
        quu, qux = R + BtP @ B, BtP @ A
        K[i] = np.linalg.solve(quu, qux)
        step[i] = -np.linalg.solve(quu, qu)
        Acl = A - B @ K[i]
        P[i] = Qc + K[i].T @ R @ K[i] + Acl.T @ P[i + 1] @ Acl
        P[i] = 0.5 * (P[i] + P[i].T)
        p = qx + qux.T @ step[i]
    return step, K, P


def _frozen_gain(A, B, Qc, R, horizon, terminal=10.0):
    """LQR gain for one linearization held over `horizon` control steps."""
    P = terminal * Qc
    for _ in range(horizon):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        Acl = A - B @ K
        P = Qc + K.T @ R @ K + Acl.T @ P @ Acl
    return K


def bootstrap(sim, ref, ff, k0=1, k1=None, horizon=25, tol=None):
    """A first real trajectory near the reference: the feedforward plus feedback on the error against the
    reference, with gains recomputed every step from a linearization at the simulator's current state (so they
    know which contacts are really closed). Returns a Teacher whose nominal is that rollout; P is not set."""
    Qr, Vr = control_states(ref)
    Qc, R = weights(sim.nv, sim.nu, tol)
    k1 = min(len(ff.u), len(Qr) - 1) if k1 is None else k1
    q, v = Qr[k0].copy(), Vr[k0].copy()
    qs, vs, us, Ks, fell = [q], [v], [], [], -1
    for k in range(k0, k1):
        A, B = sim.linearize(q, v, ff.u[k])
        K = _frozen_gain(A, B, Qc, R, horizon)
        u = np.clip(ff.u[k] - K @ sim.error(Qr[k], Vr[k], q, v), sim.lo, sim.hi)
        q, v = sim.step(q, v, u)
        qs.append(q), vs.append(v), us.append(u), Ks.append(K)
        if _terminated(q, Qr[k + 1]):
            fell = k + 1
            break
    n = 2 * sim.nv
    nom = Teacher(np.array(qs), np.array(vs), np.array(us), np.array(Ks), np.zeros((len(us) + 1, n, n)), k0,
                  fell < 0, [], fell, "bootstrap")
    if nom.feasible:
        nom.cost.append(rollout(sim, ref, nom, ff=ff, tol=tol).cost)
    return nom


def _forward(sim, ref, ff, cand, step, tol, alphas, below=np.inf):
    """Line search of one forward pass: the first step size whose cost is under `below`, else the best seen."""
    best = None
    for a in alphas:
        r = rollout(sim, ref, cand, ff=ff, tol=tol, step=step, alpha=a)
        if best is None or r.cost < best.cost:
            best = r
        if r.cost < below:
            break
    return best


def ilqr(sim, ref, ff, qpos=None, k0=1, k1=None, passes=1, tol=None, alphas=(1.0, 0.5, 0.25, 0.1), horizon=25,
         verbose=False):
    """Stage 2. qpos: seated control-step positions (seat) to linearize the first pass along; without them the
    reference itself is used, which only works if its feet are in contact. `passes` further Riccati-plus-rollout
    passes follow along the previous rollout, each kept only if it lowers the tracking cost in the simulator. If no
    first trajectory survives, the returned Teacher is not feasible and holds the bootstrap rollout up to its fall."""
    say = print if verbose else (lambda *a: None)
    nom = open_loop(ref, ff, k0, k1, qpos)
    step, K, P = _backward(sim, ref, ff, nom, tol)
    r = _forward(sim, ref, ff, Teacher(nom.q, nom.v, nom.u, K, P, nom.k0, False, []), step, tol, alphas)
    if np.isfinite(r.cost):
        say(f"  first pass: cost {r.cost:.4g}")
        nom = Teacher(r.q, r.v, r.u, K, P, nom.k0, True, [r.cost], -1, "seated")
    else:
        say(f"  first pass fell at step {r.fell}; trying the bootstrap")
        nom = bootstrap(sim, ref, ff, k0, k1, horizon, tol)
        nom.seated_fell = r.fell
        say("  bootstrap: " + (f"cost {nom.cost[0]:.4g}" if nom.feasible else f"fell at step {nom.fell}"))
        if not nom.feasible:
            return nom
    stale = True  # K and P do not yet belong to this nominal
    for it in range(passes):
        step, K, P = _backward(sim, ref, ff, nom, tol)
        cand = Teacher(nom.q, nom.v, nom.u, K, P, nom.k0, True, nom.cost, -1, nom.start)
        r = _forward(sim, ref, ff, cand, step, tol, alphas, nom.cost[-1])
        say(f"  pass {it + 1}: cost {r.cost:.4g}" + ("" if r.fell < 0 else f" (fell at step {r.fell})"))
        if r.cost >= nom.cost[-1]:
            nom, stale = cand, False  # keep the nominal, with gains and cost-to-go linearized along it
            break
        nom = Teacher(r.q, r.v, r.u, K, P, nom.k0, True, nom.cost + [r.cost], -1, nom.start)
    if stale:
        _, K, P = _backward(sim, ref, ff, nom, tol)
        nom = Teacher(nom.q, nom.v, nom.u, K, P, nom.k0, True, nom.cost, -1, nom.start)
    return nom
