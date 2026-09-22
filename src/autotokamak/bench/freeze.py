# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Solve the frozen head-to-head test set (bench freeze-testset).

Ground truth every benchmark condition is scored against. Uses the platform
sweep machinery (``data.sweep.run_sweep`` with an explicit sample matrix) on
the canonical dataset config, whose output grid IS the frozen evaluation
grid. The resulting .h5 is gitignored (heavy) but reproducible from the
committed ``benchmarks/assets/test_params.json``.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np

from autotokamak.agent.runners.config import REPO_ROOT

CANONICAL_SWEEP_CONFIG = REPO_ROOT / "examples" / "dataset_generation" / "dataset_config.yaml"


def solve_testset(
    records: list[dict],
    out_h5: Path,
    *,
    grid_json: Path | None = None,
    sweep_config: Path | None = None,
) -> int:
    """Solve ``records`` (list of param dicts) and write ``out_h5``.

    Returns the number of successful solves. Asserts the sweep config's
    output grid matches the frozen eval grid so head-to-head scoring is
    guaranteed consistent.
    """
    from autotokamak.bench.contract import grid_axes, load_grid
    from autotokamak.data.schema import SweepConfig
    from autotokamak.data.sweep import run_sweep
    from autotokamak.surrogate.dataset import PARAM_ORDER

    cfg = SweepConfig.from_yaml(sweep_config or CANONICAL_SWEEP_CONFIG)
    out_h5 = Path(out_h5)
    cfg = cfg.model_copy(update={"output_path": out_h5.name})

    R_ref, Z_ref = grid_axes(load_grid(grid_json))
    R_cfg = np.linspace(cfg.output_grid.R.min, cfg.output_grid.R.max, cfg.output_grid.R.n)
    Z_cfg = np.linspace(cfg.output_grid.Z.min, cfg.output_grid.Z.max, cfg.output_grid.Z.n)
    if not (np.allclose(R_cfg, R_ref) and np.allclose(Z_cfg, Z_ref)):
        raise ValueError(
            "Sweep config output_grid differs from the frozen eval grid — "
            "refusing to build an incomparable test set."
        )

    X = np.array([[float(r[p]) for p in PARAM_ORDER] for r in records], dtype=np.float64)
    result = run_sweep(cfg, out_h5.parent, X=X)
    stamp_provenance(out_h5, n_params=len(records))
    return int(result.n_succeeded)


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def stamp_provenance(out_h5: Path, *, n_params: int | None = None) -> dict:
    """Write provenance attrs onto the frozen test set, in place.

    The .h5 is gitignored, so the only way a published score can be tied to
    the ground truth it was computed against is a stamp inside the file
    itself. Attribute-only: the arrays are never touched, so stamping an
    existing test set does NOT invalidate prior scores (regenerating it
    silently would).
    """
    import h5py

    assets = REPO_ROOT / "benchmarks" / "assets"
    meta: dict[str, object] = {
        "stamped_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "test_params_sha256": _sha256(assets / "test_params.json") or "",
        "eval_grid_sha256": _sha256(assets / "eval_grid.json") or "",
        "sweep_config": str(CANONICAL_SWEEP_CONFIG.relative_to(REPO_ROOT)),
        "sweep_config_sha256": _sha256(CANONICAL_SWEEP_CONFIG) or "",
        # The NaN-handling rule in rel_l2_errors changed on this date; scores
        # are only poolable within one epoch.
        "scoring_epoch": "2026-08-18",
    }
    if n_params is not None:
        meta["n_params"] = int(n_params)
    try:
        from OpenFUSIONToolkit import __version__ as oft_version  # type: ignore
        meta["oft_version"] = str(oft_version)
    except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
        meta["oft_version"] = "unknown"

    with h5py.File(out_h5, "a") as f:
        for k, v in meta.items():
            f.attrs[k] = v
    return meta


def load_test_params(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
