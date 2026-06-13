"""Extract scalar diagnostics from a solved TokaMaker equilibrium.

Diagnostics here are the headline numbers used in evaluation, summaries, and
loss functions (when training surrogates):

- Toroidal current ``Ip`` (target check)
- Magnetic axis ``(R_axis, Z_axis)``
- ``q_0``, ``q_95``, ``q_edge`` from the safety-factor profile
- Pressure ``p_axis``, ``p_edge``
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _try(fn, *args, default=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return default


def extract_scalars(gs: Any) -> Dict[str, Any]:
    """Best-effort scalar dictionary; tolerant to OFT API drift across versions."""
    out: Dict[str, Any] = {}

    # Magnetic axis — different OFT builds expose this under different names.
    for cand in ("o_point", "mag_axis", "axis"):
        v = getattr(gs, cand, None)
        if callable(v):
            v = _try(v)
        if v is not None:
            try:
                arr = np.asarray(v, dtype=float).ravel()
                if arr.size >= 2:
                    out["R_axis"] = float(arr[0])
                    out["Z_axis"] = float(arr[1])
                    out["axis_source"] = cand
                    break
            except Exception:  # noqa: BLE001
                continue

    # Safety factor profile + headline values.
    q = _try(gs.get_q, psi_pad=0.005)
    if q is not None:
        try:
            psi_q, qvals = q[0], q[1]
            out["q_0"] = float(qvals[0])
            out["q_edge"] = float(qvals[-1])
            out["q_95"] = float(np.interp(0.95, psi_q, qvals))
        except Exception:  # noqa: BLE001
            pass

    # Pressure profile.
    profiles = _try(gs.get_profiles)
    if profiles is not None:
        try:
            _, _, _, p, _ = profiles  # (psi, f, fp, p, pp)
            out["p_axis"] = float(np.nanmax(np.asarray(p, dtype=float)))
            out["p_edge"] = float(np.asarray(p, dtype=float)[-1])
        except Exception:  # noqa: BLE001
            pass

    # General stats blob if available.
    stats = _try(gs.get_stats)
    if stats is not None:
        out["stats"] = stats

    return out


__all__ = ["extract_scalars"]
