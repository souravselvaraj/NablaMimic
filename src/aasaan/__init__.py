"""Aasaan: feedback-distilled trajectory optimization for humanoid motion tracking (work in progress).

Lives next to nablamimic and reuses its robot model and reference conventions, so every number is measured on the
benchmark's own G1, timestep and contact set. Nothing here imports JAX: the diagnostics run on CPU MuJoCo.

Implemented so far:
  exp0    reference feasibility diagnostics (python -m aasaan.exp0 --help)
  teach   feedforward from inverse dynamics and the iLQR teacher, with gates 1 and 2 (python -m aasaan.teach --help)
  hold    the same feedback around a logged feasible rollout instead of a reference (python -m aasaan.hold --help)
"""

__version__ = "0.0.5"
