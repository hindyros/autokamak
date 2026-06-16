#!/usr/bin/env python3
import sys
import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import yaml
import h5py

from datetime import datetime, timezone
import subprocess

from scipy.stats import qmc

import matplotlib
matplotlib.use("Agg")
import matplotlib.tri as mtri

from scipy.interpolate import griddata

from autotokamak.core.geometry import build_lcfs, build_mesh
from autotokamak.core.solver import solve_equilibrium


class ConfigError(ValueError):
    pass


def _path_join(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent else key


def _require(d: Dict[str, Any], key: str, path: str) -> Any:
    if not isinstance(d, dict):
        raise ConfigError(f"Expected mapping at '{path or '<root>'}', got {type(d).__name__}")
    if key not in d:
        raise ConfigError(f"Missing required key '{_path_join(path, key)}'")
    return d[key]


def _as_int(x: Any, path: str) -> int:
    # allow YAML ints and integer-valued floats
    if isinstance(x, bool):
        raise ConfigError(f"Expected int at '{path}', got bool")
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float) and float(x).is_integer():
        return int(x)
    raise ConfigError(f"Expected int at '{path}', got {type(x).__name__}: {x!r}")


def _as_float(x: Any, path: str) -> float:
    if isinstance(x, bool):
        raise ConfigError(f"Expected float at '{path}', got bool")
    if isinstance(x, (int, float)):
        return float(x)
    raise ConfigError(f"Expected float at '{path}', got {type(x).__name__}: {x!r}")


def _validate_bounds(param_cfg: Dict[str, Any], path: str, dtype: str) -> Tuple[float, float]:
    low = _as_float(_require(param_cfg, "low", path), _path_join(path, "low"))
    high = _as_float(_require(param_cfg, "high", path), _path_join(path, "high"))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ConfigError(f"Non-finite bounds at '{path}': low={low}, high={high}")
    if not (low < high):
        raise ConfigError(f"Invalid bounds at '{path}': require low < high, got {low} !< {high}")
    if dtype not in ("float", "int"):
        raise ConfigError(f"Invalid dtype at '{_path_join(path, 'dtype')}': {dtype!r} (expected 'float' or 'int')")
    return low, high


@dataclass
class ValidatedConfig:
    sampling_method: str
    n_samples: int
    seed: int | None
    parameters: Dict[str, Dict[str, Any]]
    fixed: Dict[str, Any]
    grid_R: np.ndarray
    grid_Z: np.ndarray
    dataset_path: str


def validate_config(cfg: Dict[str, Any]) -> ValidatedConfig:
    # top-level keys
    sampling = _require(cfg, "sampling", "")
    parameters = _require(cfg, "parameters", "")
    fixed = _require(cfg, "fixed", "")
    output_grid = _require(cfg, "output_grid", "")
    output = _require(cfg, "output", "")

    # sampling
    smethod = _require(sampling, "method", "sampling")
    if smethod not in ("lhs", "uniform"):
        raise ConfigError(f"Invalid sampling.method: {smethod!r} (expected 'lhs' or 'uniform')")
    n_samples = _as_int(_require(sampling, "n_samples", "sampling"), "sampling.n_samples")
    if n_samples <= 0:
        raise ConfigError("sampling.n_samples must be > 0")
    seed = sampling.get("seed", None)
    if seed is not None:
        seed = _as_int(seed, "sampling.seed")

    # parameters
    if not isinstance(parameters, dict) or len(parameters) == 0:
        raise ConfigError("'parameters' must be a non-empty mapping")

    # enforce required swept keys for this example
    required_params = ["r0", "a", "kappa", "delta", "Ip"]
    for p in required_params:
        if p not in parameters:
            raise ConfigError(f"Missing required swept parameter 'parameters.{p}'")

    # bounds + dtype
    for name, pcfg in parameters.items():
        ppath = f"parameters.{name}"
        if not isinstance(pcfg, dict):
            raise ConfigError(f"Expected mapping at '{ppath}', got {type(pcfg).__name__}")
        dtype = pcfg.get("dtype", "float")
        low, high = _validate_bounds(pcfg, ppath, dtype)
        # store normalized values
        pcfg["low"], pcfg["high"], pcfg["dtype"] = low, high, dtype

    # fixed
    if not isinstance(fixed, dict):
        raise ConfigError(f"Expected mapping at 'fixed', got {type(fixed).__name__}")
    z0 = _as_float(_require(fixed, "z0", "fixed"), "fixed.z0")
    F0 = _as_float(_require(fixed, "F0", "fixed"), "fixed.F0")
    npts = _as_int(_require(fixed, "npts", "fixed"), "fixed.npts")
    mesh_dx = _as_float(_require(fixed, "mesh_dx", "fixed"), "fixed.mesh_dx")
    solver_order = _as_int(_require(fixed, "solver_order", "fixed"), "fixed.solver_order")
    Ip_ratio = _as_float(_require(fixed, "Ip_ratio", "fixed"), "fixed.Ip_ratio")

    if npts <= 10:
        raise ConfigError("fixed.npts must be > 10")
    if mesh_dx <= 0:
        raise ConfigError("fixed.mesh_dx must be > 0")
    if solver_order != 1:
        raise ConfigError("This example requires fixed.solver_order == 1 so len(get_psi()) == len(mesh_pts)")
    if not (Ip_ratio > 0):
        raise ConfigError("fixed.Ip_ratio must be > 0")

    # output_grid
    if not isinstance(output_grid, dict):
        raise ConfigError(f"Expected mapping at 'output_grid', got {type(output_grid).__name__}")
    Rcfg = _require(output_grid, "R", "output_grid")
    Zcfg = _require(output_grid, "Z", "output_grid")
    for axis_name, acfg in (("R", Rcfg), ("Z", Zcfg)):
        apath = f"output_grid.{axis_name}"
        if not isinstance(acfg, dict):
            raise ConfigError(f"Expected mapping at '{apath}', got {type(acfg).__name__}")
        amin = _as_float(_require(acfg, "min", apath), f"{apath}.min")
        amax = _as_float(_require(acfg, "max", apath), f"{apath}.max")
        an = _as_int(_require(acfg, "n", apath), f"{apath}.n")
        if not (amin < amax):
            raise ConfigError(f"{apath}: require min < max")
        if an <= 1:
            raise ConfigError(f"{apath}.n must be > 1")

    R = np.linspace(float(Rcfg["min"]), float(Rcfg["max"]), int(Rcfg["n"]), dtype=np.float64)
    Z = np.linspace(float(Zcfg["min"]), float(Zcfg["max"]), int(Zcfg["n"]), dtype=np.float64)

    # output
    dataset_path = _require(output, "dataset_path", "output")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise ConfigError("output.dataset_path must be a non-empty string")

    fixed_norm = {
        "z0": z0,
        "F0": F0,
        "npts": npts,
        "mesh_dx": mesh_dx,
        "solver_order": solver_order,
        "Ip_ratio": Ip_ratio,
    }

    return ValidatedConfig(
        sampling_method=smethod,
        n_samples=n_samples,
        seed=seed,
        parameters=parameters,
        fixed=fixed_norm,
        grid_R=R,
        grid_Z=Z,
        dataset_path=dataset_path,
    )


def sample_parameters(vcfg: ValidatedConfig) -> Dict[str, np.ndarray]:
    """Sample swept parameters.

    Determinism/reproducibility
    ---------------------------
    * For ``sampling.method == 'lhs'`` we use ``scipy.stats.qmc.LatinHypercube``
      with ``seed=...``.
    * Parameters are sampled in a stable, config-defined order: the insertion
      order of the YAML ``parameters`` mapping.

    Scaling
    -------
    Unit-hypercube samples ``u in [0,1)^d`` are mapped to bounds via
    ``qmc.scale(u, lows, highs)``.

    Integer parameters
    ------------------
    If a swept parameter is declared with ``dtype: int`` then we map the scaled
    value ``x`` to an integer using ``floor(x)`` and clip to the inclusive
    integer range ``[ceil(low), floor(high)]``.

    Note: this dataset example only sweeps float parameters, but the logic is
    included for completeness.
    """
    names = list(vcfg.parameters.keys())
    lows = np.array([vcfg.parameters[n]["low"] for n in names], dtype=float)
    highs = np.array([vcfg.parameters[n]["high"] for n in names], dtype=float)
    dtypes = [vcfg.parameters[n]["dtype"] for n in names]

    if vcfg.sampling_method == "lhs":
        engine = qmc.LatinHypercube(d=len(names), seed=vcfg.seed)
        u = engine.random(n=vcfg.n_samples)  # u in [0,1)
        x = qmc.scale(u, lows, highs)
    else:
        rng = np.random.default_rng(vcfg.seed)
        x = rng.uniform(lows, highs, size=(vcfg.n_samples, len(names)))

    out: Dict[str, np.ndarray] = {}
    for j, n in enumerate(names):
        if dtypes[j] == "int":
            ij_min = int(np.ceil(lows[j]))
            ij_max = int(np.floor(highs[j]))
            if ij_min > ij_max:
                raise ConfigError(
                    f"Integer bounds for '{n}' contain no integers: low={lows[j]}, high={highs[j]}"
                )
            vals = np.floor(x[:, j]).astype(np.int64)
            vals = np.clip(vals, ij_min, ij_max)
            out[n] = vals
        else:
            out[n] = x[:, j].astype(np.float64)
    return out


def interpolate_psi_to_grid(*, mesh_pts: np.ndarray, mesh_lc: np.ndarray, psi_nodes: np.ndarray,
                            R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Interpolate psi defined at mesh nodes onto a rectilinear (R,Z) grid.

    Primary method: triangulation + LinearTriInterpolator using provided mesh triangles.
    Fallback: scipy.interpolate.griddata(linear) with nearest fill for any remaining NaNs.

    Returns
    -------
    psi_grid : (nz, nr) float64, indexed as psi_grid[z_index, r_index]
    """
    if mesh_pts.ndim != 2 or mesh_pts.shape[1] != 2:
        raise ValueError(f"mesh_pts must have shape (N,2); got {mesh_pts.shape}")
    if psi_nodes.shape[0] != mesh_pts.shape[0]:
        raise ValueError(f"psi_nodes length {psi_nodes.shape[0]} != mesh_pts nodes {mesh_pts.shape[0]}")

    RR, ZZ = np.meshgrid(R, Z, indexing="xy")  # (nz, nr)

    # Try mesh-connectivity-based triangulation first (no Delaunay inference).
    try:
        tri = mtri.Triangulation(mesh_pts[:, 0], mesh_pts[:, 1], triangles=mesh_lc)
        interp = mtri.LinearTriInterpolator(tri, psi_nodes)
        psi_grid = np.asarray(interp(RR, ZZ).filled(np.nan), dtype=np.float64)
    except Exception:
        psi_grid = np.full(RR.shape, np.nan, dtype=np.float64)

    # Fallback if the tri interpolator failed entirely or left gaps.
    if (not np.isfinite(psi_grid).any()) or np.isnan(psi_grid).any():
        points = np.asarray(mesh_pts, dtype=np.float64)
        values = np.asarray(psi_nodes, dtype=np.float64)
        psi_lin = griddata(points, values, (RR, ZZ), method="linear")
        if psi_lin is None:
            psi_lin = np.full(RR.shape, np.nan, dtype=np.float64)
        psi_lin = np.asarray(psi_lin, dtype=np.float64)
        if np.isnan(psi_lin).any():
            psi_near = griddata(points, values, (RR, ZZ), method="nearest")
            psi_near = np.asarray(psi_near, dtype=np.float64)
            psi_lin = np.where(np.isnan(psi_lin), psi_near, psi_lin)
        # If tri method produced some valid values, prefer them.
        psi_grid = np.where(np.isfinite(psi_grid), psi_grid, psi_lin)

    return psi_grid.astype(np.float64, copy=False)


def _try_git_commit() -> str | None:
    try:
        cp = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if cp.returncode == 0:
            sha = (cp.stdout or "").strip()
            return sha or None
    except Exception:
        return None
    return None


def preflight() -> None:
    """Lightweight environment sanity checks.

    This is intentionally minimal: fail fast with a clear error if the runtime
    is missing core dependencies.
    """
    if sys.version_info < (3, 11):
        raise RuntimeError(f"Python >= 3.11 required; found {sys.version.split()[0]}")

    required = [
        ("yaml", "PyYAML"),
        ("h5py", "h5py"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("autotokamak", "autotokamak"),
    ]
    import importlib

    missing: list[str] = []
    for mod, pipname in required:
        try:
            importlib.import_module(mod)
        except Exception as e:
            missing.append(f"{pipname} (import '{mod}' failed: {e})")

    if missing:
        msg = "Missing/failed imports:\n  - " + "\n  - ".join(missing)
        raise RuntimeError(msg)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python run_dataset_sweep.py <config.yaml>", file=sys.stderr)
        return 2

    try:
        preflight()
    except Exception as e:
        print(f"Preflight failed: {e}", file=sys.stderr)
        return 2

    cfg_path = argv[1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_text = f.read()
        cfg = yaml.safe_load(cfg_text)

    try:
        vcfg = validate_config(cfg)
    except Exception as e:
        print(f"Config validation failed: {e}", file=sys.stderr)
        return 2

    R = vcfg.grid_R
    Z = vcfg.grid_Z
    nr, nz = len(R), len(Z)

    samples = sample_parameters(vcfg)
    N = vcfg.n_samples

    psi_out = np.full((N, nz, nr), np.nan, dtype=np.float64)
    success = np.zeros((N,), dtype=bool)

    # input arrays (float64 as requested)
    in_r0 = np.full((N,), np.nan, dtype=np.float64)
    in_a = np.full((N,), np.nan, dtype=np.float64)
    in_kappa = np.full((N,), np.nan, dtype=np.float64)
    in_delta = np.full((N,), np.nan, dtype=np.float64)
    in_Ip = np.full((N,), np.nan, dtype=np.float64)

    for i in range(N):
        r0 = float(samples["r0"][i])
        a = float(samples["a"][i])
        kappa = float(samples["kappa"][i])
        delta = float(samples["delta"][i])
        Ip = float(samples["Ip"][i])

        in_r0[i], in_a[i], in_kappa[i], in_delta[i], in_Ip[i] = r0, a, kappa, delta, Ip

        try:
            lcfs = build_lcfs(r0=r0, z0=vcfg.fixed["z0"], a=a, kappa=kappa, delta=delta, npts=vcfg.fixed["npts"])
            mesh, mesh_pts, mesh_lc, mesh_reg = build_mesh(lcfs, mesh_dx=vcfg.fixed["mesh_dx"], region_name="plasma", region_tag="plasma")

            solve_cfg = {
                "equation": {"name": "gs"},
                "boundary": {
                    "type": "isoflux",
                    "r0": r0,
                    "z0": vcfg.fixed["z0"],
                    "a": a,
                    "kappa": kappa,
                    "delta": delta,
                    "npts": vcfg.fixed["npts"],
                },
                "mesh": {
                    "method": "gs_domain",
                    "regions": [{"name": "plasma", "type": "plasma", "dx": vcfg.fixed["mesh_dx"]}],
                },
                "solver": {"order": vcfg.fixed["solver_order"], "F0": vcfg.fixed["F0"], "free_boundary": False},
                "targets": {"Ip": Ip, "Ip_ratio": vcfg.fixed["Ip_ratio"]},
                "init_psi": {"method": "tokamaker_default"},
            }

            gs = solve_equilibrium(mesh_pts=mesh_pts, mesh_lc=mesh_lc, mesh_reg=mesh_reg, lcfs=lcfs, cfg=solve_cfg)
            psi_nodes = np.asarray(gs.get_psi(), dtype=float).ravel()
            if len(psi_nodes) != len(mesh_pts):
                raise RuntimeError(f"len(get_psi())={len(psi_nodes)} != len(mesh_pts)={len(mesh_pts)}; ensure solver.order == 1")

            psi_grid = interpolate_psi_to_grid(mesh_pts=mesh_pts, mesh_lc=mesh_lc, psi_nodes=psi_nodes, R=R, Z=Z)

            psi_out[i, :, :] = psi_grid
            success[i] = True
        except Exception as e:
            print(f"Sample {i} failed: {e}")
            traceback.print_exc(limit=2)
            success[i] = False
            # psi_out already NaN

    out_path = vcfg.dataset_path
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    compression = "gzip"
    compression_opts = 4

    # Chunk per-sample slabs for efficient reading: (1, nz, nr)
    psi_chunks = (1, nz, nr)

    with h5py.File(out_path, "w") as h5:
        # -------- metadata (root attrs + small datasets) --------
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["generator"] = "run_dataset_sweep.py"
        h5.attrs["equilibrium"] = "fixed-boundary Grad-Shafranov (autotokamak.core + OpenFUSIONToolkit/TokaMaker)"
        h5.attrs["axis_order_outputs/psi"] = "(sample, z, r)"
        h5.attrs["interp_primary"] = "matplotlib.tri.LinearTriInterpolator"
        h5.attrs["interp_fallback"] = "scipy.interpolate.griddata (linear then nearest fill)"
        h5.attrs["dtype/psi"] = "float64"
        h5.attrs["dtype/inputs"] = "float64"
        h5.attrs["units/R"] = "m"
        h5.attrs["units/Z"] = "m"
        h5.attrs["units/r0"] = "m"
        h5.attrs["units/a"] = "m"
        h5.attrs["units/kappa"] = "dimensionless"
        h5.attrs["units/delta"] = "dimensionless"
        h5.attrs["units/Ip"] = "A"

        git_sha = _try_git_commit()
        if git_sha is not None:
            h5.attrs["git_commit"] = git_sha

        # store full config snapshot for provenance
        h5.create_dataset("config_yaml", data=np.bytes_(cfg_text))

        # parameter names/order used in sampling and stored arrays
        param_names = ["r0", "a", "kappa", "delta", "Ip"]
        h5.create_dataset("parameter_names", data=np.array(param_names, dtype=h5py.string_dtype("utf-8")))

        # store swept bounds as (n_params, 2)
        bounds = np.array([
            [vcfg.parameters["r0"]["low"], vcfg.parameters["r0"]["high"]],
            [vcfg.parameters["a"]["low"], vcfg.parameters["a"]["high"]],
            [vcfg.parameters["kappa"]["low"], vcfg.parameters["kappa"]["high"]],
            [vcfg.parameters["delta"]["low"], vcfg.parameters["delta"]["high"]],
            [vcfg.parameters["Ip"]["low"], vcfg.parameters["Ip"]["high"]],
        ], dtype=np.float64)
        db = h5.create_dataset("parameter_bounds", data=bounds)
        db.attrs["columns"] = np.array(["low", "high"], dtype=h5py.string_dtype("utf-8"))

        # Also store the sampled parameter matrix for convenience: (N, n_params)
        X = np.stack([in_r0, in_a, in_kappa, in_delta, in_Ip], axis=1).astype(np.float64)
        dX = h5.create_dataset("inputs_matrix", data=X, compression=compression, compression_opts=compression_opts, chunks=(min(1024, N), len(param_names)))
        dX.attrs["axis_order"] = "(sample, parameter)"

        # -------- main data groups (original minimal layout) --------
        ggrid = h5.create_group("grid")
        dR = ggrid.create_dataset("R", data=R)
        dZ = ggrid.create_dataset("Z", data=Z)
        dR.attrs["units"] = "m"
        dZ.attrs["units"] = "m"

        gins = h5.create_group("inputs")
        gins.create_dataset("r0", data=in_r0, compression=compression, compression_opts=compression_opts, chunks=True)
        gins.create_dataset("a", data=in_a, compression=compression, compression_opts=compression_opts, chunks=True)
        gins.create_dataset("kappa", data=in_kappa, compression=compression, compression_opts=compression_opts, chunks=True)
        gins.create_dataset("delta", data=in_delta, compression=compression, compression_opts=compression_opts, chunks=True)
        gins.create_dataset("Ip", data=in_Ip, compression=compression, compression_opts=compression_opts, chunks=True)

        gouts = h5.create_group("outputs")
        dpsi = gouts.create_dataset(
            "psi",
            data=psi_out,
            dtype=np.float64,
            compression=compression,
            compression_opts=compression_opts,
            chunks=psi_chunks,
            shuffle=True,
        )
        dpsi.attrs["axis_order"] = "(sample, z, r)"
        dpsi.attrs["convention"] = "psi(R,Z) on rectilinear grid; NaN outside interpolation domain"

        gouts.create_dataset("success", data=success, compression=compression, compression_opts=compression_opts, chunks=True)

    M = int(success.sum())
    print(f"Wrote {os.path.basename(out_path)}: {M}/{N} succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
