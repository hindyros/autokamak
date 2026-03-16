#!/usr/bin/env python3
"""Config-driven OpenFUSIONToolkit (OFT) / TokaMaker equilibrium example.

Implements a single primary workflow:
- equation.type == "grad_shafranov"
- mesh.method  == "gs_domain" (single plasma region)
- geometry.lcfs.type == "create_isoflux"

Reads all physics + discretization from the config; no case-specific hardcoding.

Definition-of-done metrics:
- solver must report a successful solve call (no exception)
- optional numeric checks (configurable):
  - min_psi_finite: psi(normalized) must be finite on all nodes
  - targets: compare achieved vs requested if available in get_stats()

The runner records these metrics in outputs/<case>/summary.yaml and exits nonzero
if checks fail (unless checks.allow_fail: true).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required (should be available from requirements.txt)") from e


class ConfigError(ValueError):
    pass


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _to_float(x: Any, path: str) -> float:
    if not _is_number(x):
        raise ConfigError(f"Expected numeric at {path}, got {type(x).__name__}: {x!r}")
    return float(x)


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(text) or {}
    if path.suffix.lower() == ".json":
        return json.loads(text)
    raise ConfigError(f"Unsupported config extension: {path.suffix}")


def _to_builtin(obj: Any) -> Any:
    """Convert numpy/scalars to plain Python types so YAML/JSON can serialize."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def _dump_yaml(path: Path, obj: Any) -> None:
    path.write_text(yaml.safe_dump(_to_builtin(obj), sort_keys=False))


def _recursive_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _recursive_merge(out[k], v)
        else:
            out[k] = v
    return out


def _defaults() -> Dict[str, Any]:
    return {
        "metadata": {"name": "oft_case"},
        "io": {"output_dir": "outputs", "save_plots": True},
        "equation": {"type": "grad_shafranov"},
        "mesh": {"method": "gs_domain", "regions": {"plasma": {"dx": 0.05}}},
        "solver": {"fe_order": 2, "settings": {}},
        "profiles": {"use_defaults": True},
        "checks": {
            # If true, do not exit nonzero when checks fail.
            "allow_fail": False,
            # If true, require psi_hat to exist and be finite on all nodes.
            "require_finite_psi": True,
            # Target tolerance checks (only if achieved values can be found in stats)
            "targets": {
                "enabled": True,
                "rtol": 5.0e-2,
                "atol": 0.0,
            },
        },
        "postprocess": {
            "export": {
                "write_native_mesh": True,
                "write_summary": True,
                "write_psi_on_nodes": True,
            },
            "plots": {"mesh": True, "psi_contours": True},
        },
    }


def _validate(cfg: Dict[str, Any]) -> None:
    if cfg.get("equation", {}).get("type") != "grad_shafranov":
        raise ConfigError('Only equation.type="grad_shafranov" is implemented in this example runner')
    if cfg.get("mesh", {}).get("method") != "gs_domain":
        raise ConfigError('Only mesh.method="gs_domain" is implemented in this example runner')

    lcfs = cfg.get("geometry", {}).get("lcfs", {})
    if lcfs.get("type") != "create_isoflux":
        raise ConfigError('Only geometry.lcfs.type="create_isoflux" is implemented')

    for key in ["r0", "z0", "a", "kappa", "delta", "n_pts"]:
        if key not in lcfs:
            raise ConfigError(f"Missing geometry.lcfs.{key}")

    dx = cfg.get("mesh", {}).get("regions", {}).get("plasma", {}).get("dx")
    dx = _to_float(dx, "mesh.regions.plasma.dx")
    if dx <= 0:
        raise ConfigError("mesh.regions.plasma.dx must be > 0")

    phys = cfg.get("physics", {})
    # Physics: F0 required; targets may be disabled by setting to OFT_env.float_disable_flag
    for k in ["F0", "Ip", "Ip_ratio"]:
        if k not in phys:
            raise ConfigError(f"Missing physics.{k}")
        _to_float(phys[k], f"physics.{k}")

    # checks
    chk = cfg.get("checks", {})
    if "targets" in chk and chk["targets"].get("enabled", False):
        _to_float(chk["targets"].get("rtol", 0.0), "checks.targets.rtol")
        _to_float(chk["targets"].get("atol", 0.0), "checks.targets.atol")


@dataclass
class CasePaths:
    case_dir: Path
    fig_dir: Path
    data_dir: Path


def _make_case_dirs(cfg: Dict[str, Any], config_path: Path) -> CasePaths:
    out_root = Path(cfg["io"]["output_dir"]).expanduser().resolve()
    case_name = cfg["metadata"]["name"]
    case_dir = out_root / case_name
    fig_dir = case_dir / "figures"
    data_dir = case_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    _dump_yaml(case_dir / f"{config_path.stem}.resolved.yaml", cfg)
    return CasePaths(case_dir=case_dir, fig_dir=fig_dir, data_dir=data_dir)


def _build_mesh_from_gs_domain(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
    from OpenFUSIONToolkit.TokaMaker.util import create_isoflux

    lcfs = cfg["geometry"]["lcfs"]
    npts = int(lcfs["n_pts"])
    r0 = _to_float(lcfs["r0"], "geometry.lcfs.r0")
    z0 = _to_float(lcfs["z0"], "geometry.lcfs.z0")
    a = _to_float(lcfs["a"], "geometry.lcfs.a")
    kappa = _to_float(lcfs["kappa"], "geometry.lcfs.kappa")
    delta = _to_float(lcfs["delta"], "geometry.lcfs.delta")

    dx = _to_float(cfg["mesh"]["regions"]["plasma"]["dx"], "mesh.regions.plasma.dx")

    lcfs_pts = create_isoflux(npts, r0, z0, a, kappa, delta)
    # create_isoflux returns a contour suitable for add_polygon; no explicit closing needed.

    dom = gs_Domain()
    dom.define_region("plasma", dx=dx, reg_type="plasma")
    dom.add_polygon(lcfs_pts, "plasma")
    r, lc, reg = dom.build_mesh()

    return r, lc, reg, lcfs_pts


def _apply_profiles(mygs, cfg: Dict[str, Any]) -> None:
    prof = cfg.get("profiles", {})
    if prof.get("use_defaults", True):
        return

    pp_prof = prof.get("pp_prof")
    ffp_prof = prof.get("ffp_prof")
    foffset = prof.get("foffset")

    if pp_prof is None or ffp_prof is None:
        raise ConfigError("profiles.use_defaults=false requires profiles.pp_prof and profiles.ffp_prof")

    kwargs = {"pp_prof": pp_prof, "ffp_prof": ffp_prof}
    if foffset is not None:
        kwargs["foffset"] = float(foffset)
    mygs.set_profiles(**kwargs)


def _apply_settings(mygs, cfg: Dict[str, Any]) -> None:
    settings_patch = cfg.get("solver", {}).get("settings", {}) or {}
    for k, v in settings_patch.items():
        if not hasattr(mygs.settings, k):
            raise ConfigError(f"solver.settings.{k} is not a valid TokaMaker setting attribute")
        if isinstance(v, bool):
            setattr(mygs.settings, k, bool(v))
        elif isinstance(v, int):
            setattr(mygs.settings, k, int(v))
        elif _is_number(v):
            setattr(mygs.settings, k, float(v))
        else:
            raise ConfigError(f"Unsupported type for solver.settings.{k}: {type(v).__name__}")
    mygs.update_settings()


def _find_first_number(d: Any, keys: Tuple[str, ...]) -> float | None:
    """Search nested dict-like structures for the first numeric value under any key in keys."""
    if isinstance(d, dict):
        for k, v in d.items():
            if str(k).lower() in keys and _is_number(v):
                return float(v)
            found = _find_first_number(v, keys)
            if found is not None:
                return found
    elif isinstance(d, (list, tuple)):
        for v in d:
            found = _find_first_number(v, keys)
            if found is not None:
                return found
    return None


def _check_targets(cfg: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort target satisfaction checks based on whatever get_stats() exposes."""
    chk = cfg.get("checks", {}).get("targets", {})
    enabled = bool(chk.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    rtol = float(chk.get("rtol", 0.0))
    atol = float(chk.get("atol", 0.0))

    Ip_req = _to_float(cfg["physics"]["Ip"], "physics.Ip")

    # Try common names; API can differ across OFT versions/builds.
    Ip_ach = _find_first_number(stats, ("ip", "plasma_current", "i_p"))

    out: Dict[str, Any] = {"enabled": True, "rtol": rtol, "atol": atol}
    if Ip_ach is None:
        out.update({"available": False, "reason": "Ip not found in stats"})
        return out

    err = abs(Ip_ach - Ip_req)
    tol = atol + rtol * abs(Ip_req)
    out.update(
        {
            "available": True,
            "Ip_requested": Ip_req,
            "Ip_achieved": Ip_ach,
            "abs_error": err,
            "tolerance": tol,
            "pass": bool(err <= tol),
        }
    )
    return out


def _solve(cfg: Dict[str, Any], paths: CasePaths) -> Dict[str, Any]:
    from OpenFUSIONToolkit._core import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker

    r, lc, reg, lcfs_closed = _build_mesh_from_gs_domain(cfg)

    if cfg["postprocess"]["export"].get("write_native_mesh", True):
        from OpenFUSIONToolkit.util import write_native_mesh

        r3 = np.zeros((r.shape[0], 3), dtype=float)
        r3[:, :2] = r
        mesh_file = paths.data_dir / "mesh_native.h5"
        write_native_mesh(str(mesh_file), r3, lc + 1, reg)

    env = OFT_env(debug_level=0, nthreads=int(os.environ.get("OFT_NUM_THREADS", "2")))

    mygs = TokaMaker(env)
    mygs.setup_mesh(r=r, lc=lc, reg=reg)
    mygs.settings.free_boundary = False  # fixed-boundary (prescribed LCFS), as in docs

    fe_order = int(cfg.get("solver", {}).get("fe_order", 2))
    F0 = _to_float(cfg["physics"]["F0"], "physics.F0")
    mygs.setup(order=fe_order, F0=F0)

    _apply_settings(mygs, cfg)
    _apply_profiles(mygs, cfg)

    # Isoflux constraints (optional, config-driven)
    isoflux_cfg = cfg.get("physics", {}).get("isoflux", {})
    if bool(isoflux_cfg.get("enabled", True)):
        frac = float(isoflux_cfg.get("frac", 1.0))
        use_n = max(4, int(round(frac * lcfs_closed.shape[0])))
        use_n = min(use_n, lcfs_closed.shape[0])
        mygs.set_isoflux(lcfs_closed[:use_n, :])
    else:
        mygs.set_isoflux(None)

    Ip = _to_float(cfg["physics"]["Ip"], "physics.Ip")
    Ip_ratio = _to_float(cfg["physics"]["Ip_ratio"], "physics.Ip_ratio")
    mygs.set_targets(Ip=Ip, Ip_ratio=Ip_ratio)

    mygs.init_psi()  # as in docs (initial flux before solve)

    solve_ok = False
    err = None
    try:
        mygs.solve()
        solve_ok = True
    except Exception as e:
        err = str(e)

    stats: Dict[str, Any] = {}
    try:
        stats = mygs.get_stats() or {}
    except Exception:
        pass

    psi_finite = None
    psi_min = None
    psi_max = None
    psi_hat = None
    if cfg.get("checks", {}).get("require_finite_psi", True) or cfg["postprocess"]["export"].get(
        "write_psi_on_nodes", True
    ):
        try:
            psi_hat = mygs.get_psi(normalized=True)
            psi_finite = bool(np.isfinite(psi_hat).all())
            psi_min = float(np.nanmin(psi_hat))
            psi_max = float(np.nanmax(psi_hat))
        except Exception:
            psi_finite = False

    if cfg["postprocess"]["export"].get("write_psi_on_nodes", True) and psi_hat is not None:
        try:
            np.save(paths.data_dir / "psi_normalized_on_nodes.npy", psi_hat)
        except Exception:
            pass

    # Plots (best effort)
    if cfg.get("io", {}).get("save_plots", True):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.tri import Triangulation

            if cfg["postprocess"]["plots"].get("mesh", True):
                fig, ax = plt.subplots(figsize=(6, 6))
                tri = Triangulation(r[:, 0], r[:, 1], triangles=lc)
                ax.triplot(tri, lw=0.3, color="0.2")
                ax.plot(lcfs_closed[:, 0], lcfs_closed[:, 1], "r-", lw=1.5, label="LCFS")
                ax.set_aspect("equal")
                ax.set_xlabel("R [m]")
                ax.set_ylabel("Z [m]")
                ax.set_title("Mesh + LCFS")
                ax.legend(loc="best")
                fig.tight_layout()
                fig.savefig(paths.fig_dir / "mesh.png", dpi=200)
                plt.close(fig)

            if cfg["postprocess"]["plots"].get("psi_contours", True) and psi_finite:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                mygs.plot_psi(fig, ax)
                ax.plot(lcfs_closed[:, 0], lcfs_closed[:, 1], "r-", lw=1.0, label="LCFS")
                ax.set_aspect("equal")
                fig.tight_layout()
                fig.savefig(paths.fig_dir / "psi.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
        except Exception as e:
            err = (err + f" | plotting failed: {e}") if err else f"plotting failed: {e}"

    # Checks
    checks: Dict[str, Any] = {"solve_ok": bool(solve_ok)}
    if psi_finite is not None:
        checks["psi_hat_finite"] = bool(psi_finite)
        checks["psi_hat_min"] = psi_min
        checks["psi_hat_max"] = psi_max

    checks["targets"] = _check_targets(cfg, stats)

    # Overall pass/fail
    require_finite_psi = bool(cfg.get("checks", {}).get("require_finite_psi", True))
    targets_enabled = bool(cfg.get("checks", {}).get("targets", {}).get("enabled", False))

    pass_flags = [bool(solve_ok)]
    if require_finite_psi:
        pass_flags.append(bool(psi_finite))
    if targets_enabled and checks["targets"].get("available", False):
        pass_flags.append(bool(checks["targets"].get("pass", False)))

    checks["pass"] = bool(all(pass_flags))

    result = {
        "converged": bool(solve_ok),
        "error": err,
        "targets": {"Ip": Ip, "Ip_ratio": Ip_ratio, "F0": F0},
        "mesh": {"np": int(r.shape[0]), "nc": int(lc.shape[0])},
        "stats": stats,
        "checks": checks,
    }

    if cfg["postprocess"]["export"].get("write_summary", True):
        _dump_yaml(paths.case_dir / "summary.yaml", result)

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    config_path = Path(args.config)
    user_cfg = _load_config(config_path)
    cfg = _recursive_merge(_defaults(), user_cfg)
    _validate(cfg)

    paths = _make_case_dirs(cfg, config_path)

    result = _solve(cfg, paths)
    print("\n=== DONE ===")
    print(yaml.safe_dump(_to_builtin(result), sort_keys=False))

    allow_fail = bool(cfg.get("checks", {}).get("allow_fail", False))
    if (not result.get("checks", {}).get("pass", False)) and (not allow_fail):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
