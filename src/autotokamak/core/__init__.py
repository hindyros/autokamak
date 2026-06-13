"""Core utilities shared across examples, surrogate training, and agentic runners.

Submodules:
- :mod:`autotokamak.core.geometry`    — LCFS construction and meshing
- :mod:`autotokamak.core.solver`      — TokaMaker setup and solve (with retry-on-isoflux-fail)
- :mod:`autotokamak.core.io`          — atomic NPZ/JSON writers and unified output paths
- :mod:`autotokamak.core.diagnostics` — extract scalar diagnostics from a solved equilibrium
- :mod:`autotokamak.core.logging`     — phase/elapsed/kv terminal logger
- :mod:`autotokamak.core.schema`      — Pydantic config models (Phase R3)
"""

from autotokamak.core import diagnostics, geometry, io, logging, solver

__all__ = ["diagnostics", "geometry", "io", "logging", "solver"]
