"""TokaMaker solver setup + solve, with the retry-on-isoflux-fail fallback.

Both example runners pre-refactor implemented this dance themselves. The logic
is delicate: ``set_isoflux()`` can fail at construction time on some
mesh/shape combinations, so we keep the proven try/retry pattern from
``config_driven_equilibrium/run_equilibrium_from_config.py:setup_and_solve``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np


# OFT only permits one OFT_env per Python kernel. Cache the first one we make
# and reuse it across calls so make_solver / solve_equilibrium can be invoked
# inside a sweep loop without hitting "Only one instance of `OFT_env`...".
_OFT_ENV_CACHE: Any = None


def get_oft_env() -> Any:
    """Return a process-wide OFT_env, creating it on first call."""
    global _OFT_ENV_CACHE
    if _OFT_ENV_CACHE is None:
        import OpenFUSIONToolkit as oft  # imported lazily to avoid hard dep at import time
        _OFT_ENV_CACHE = oft.OFT_env(nthreads=int(os.getenv("OFT_NTHREADS", "2")))
    return _OFT_ENV_CACHE


def _targets_kwargs_from_cfg(cfg: Dict[str, Any]) -> Dict[str, float]:
    """Extract the kwargs to forward to ``TokaMaker.set_targets``."""
    t = cfg.get("targets", {}) or {}
    return {
        key: float(t[key])
        for key in ("Ip", "Ip_ratio", "pax", "estore", "R0", "V0")
        if key in t
    }


def _solver_setup_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sol = cfg["solver"]
    return {
        "order": int(sol["order"]),
        "F0": float(sol["F0"]),
        "full_domain": bool(sol.get("full_domain", False)),
    }


def _apply_solver_settings(gs: Any, cfg: Dict[str, Any]) -> None:
    sol = cfg["solver"]
    gs.settings.free_boundary = bool(sol.get("free_boundary", False))
    if "maxits" in sol:
        gs.settings.maxits = int(sol["maxits"])


def _seed_psi(gs: Any, cfg: Dict[str, Any]) -> None:
    """Run ``init_psi`` per the config's ``init_psi.method``, with safe fallback."""
    init = cfg.get("init_psi", {}) or {}
    method = init.get("method", "tokamaker_default")
    try:
        if method == "isoflux":
            b = cfg["boundary"]
            gs.init_psi(
                float(b["r0"]),
                float(b["z0"]),
                float(b["a"]),
                float(b["kappa"]),
                float(b["delta"]),
            )
        elif method == "tokamaker_default":
            gs.init_psi()
        else:
            raise ValueError(
                f"init_psi.method must be 'tokamaker_default' or 'isoflux', got {method!r}"
            )
    except Exception:  # noqa: BLE001
        # The internal isoflux fit used during init_psi can fail; fall back to a
        # uniform-current seed (which is what init_psi(-1.0) does in OFT).
        gs.init_psi(-1.0)


def make_solver(
    *,
    mesh_pts: np.ndarray,
    mesh_lc: np.ndarray,
    mesh_reg: np.ndarray | None,
    cfg: Dict[str, Any],
    env: Any | None = None,
) -> Tuple[Any, Any]:
    """Build a ``TokaMaker`` on an OFT env, load the mesh, apply settings & targets.

    OFT has a hard constraint: only ONE ``OFT_env`` can ever be created per Python
    kernel. So if you need a "fresh" solver (e.g. for a retry), pass the existing
    ``env`` in rather than calling this with ``env=None``.

    Returns
    -------
    (env, gs)
        The OFT environment handle (reuse this for any retries) and the
        configured-but-unsolved TokaMaker instance.
    """
    from OpenFUSIONToolkit.TokaMaker import TokaMaker

    if env is None:
        env = get_oft_env()
    gs = TokaMaker(env)
    gs.setup_mesh(mesh_pts, mesh_lc, reg=mesh_reg)
    _apply_solver_settings(gs, cfg)
    gs.setup(**_solver_setup_kwargs(cfg))
    gs.set_targets(**_targets_kwargs_from_cfg(cfg))
    return env, gs


def solve_equilibrium(
    *,
    mesh_pts: np.ndarray,
    mesh_lc: np.ndarray,
    mesh_reg: np.ndarray | None,
    lcfs: np.ndarray,
    cfg: Dict[str, Any],
) -> Any:
    """End-to-end: create solver → seed psi → set isoflux constraint → solve.

    If the isoflux-constrained solve fails (some OFT builds choke on certain
    mesh/shaping combinations), we rebuild a fresh ``TokaMaker`` on the *same*
    OFT env (kernel-level singleton) and solve without the constraint, with a
    loud warning. This preserves the exact retry behaviour from pre-refactor.

    Returns the solved TokaMaker instance.
    """
    env, gs = make_solver(mesh_pts=mesh_pts, mesh_lc=mesh_lc, mesh_reg=mesh_reg, cfg=cfg)
    _seed_psi(gs, cfg)

    try:
        gs.set_isoflux(np.asarray(lcfs, dtype=float))
        gs.solve()
        return gs
    except Exception as e:  # noqa: BLE001
        print(
            f"WARNING: set_isoflux/solve failed ({e}). Falling back to unconstrained solve."
        )

    # Retry path: reuse the existing OFT env (cannot create a new one in the
    # same kernel) and build a fresh TokaMaker on it without the isoflux constraint.
    _, gs2 = make_solver(mesh_pts=mesh_pts, mesh_lc=mesh_lc, mesh_reg=mesh_reg, cfg=cfg, env=env)
    _seed_psi(gs2, cfg)
    gs2.solve()
    return gs2


__all__ = ["get_oft_env", "make_solver", "solve_equilibrium"]
