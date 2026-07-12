"""Shared test fixtures/helpers for the autotokamak suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_synthetic_h5(
    path: Path,
    *,
    n: int = 16,
    nz: int = 8,
    nr: int = 6,
    seed: int = 0,
    n_failures: int = 0,
) -> Path:
    """Write a tiny synthetic dataset in the canonical Phase-1 HDF5 layout.

    Low-rank psi (rank 2 + noise) with a NaN border, mirroring
    ``test_surrogate_smoke._synthetic_bundle`` but persisted to disk so
    ``h5io`` / ``load_dataset`` / ``automl_loop`` round-trips can run on it.
    The last ``n_failures`` rows are marked ``success=False`` with NaN psi.
    """
    import h5py

    rng = np.random.default_rng(seed)
    R = np.linspace(0.2, 0.7, nr)
    Z = np.linspace(-0.3, 0.3, nz)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")

    basis0 = np.exp(-((RR - 0.4) ** 2 + ZZ**2) / 0.05)
    basis1 = np.exp(-((RR - 0.55) ** 2 + ZZ**2) / 0.03)

    inputs = rng.uniform(size=(n, 5))
    psi = (
        inputs[:, 0][:, None, None] * basis0[None, :, :]
        + inputs[:, 1][:, None, None] * basis1[None, :, :]
    )
    psi += rng.normal(scale=1e-3, size=psi.shape)
    psi[:, 0, :] = np.nan
    psi[:, -1, :] = np.nan
    psi[:, :, 0] = np.nan
    psi[:, :, -1] = np.nan

    success = np.ones(n, dtype=bool)
    if n_failures:
        success[n - n_failures :] = False
        psi[n - n_failures :] = np.nan
    isoflux_used = np.zeros(n, dtype=bool)

    param_names = ("r0", "a", "kappa", "delta", "Ip")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g_grid = f.create_group("grid")
        g_grid.create_dataset("R", data=R, dtype="f8")
        g_grid.create_dataset("Z", data=Z, dtype="f8")
        g_in = f.create_group("inputs")
        for j, p in enumerate(param_names):
            g_in.create_dataset(p, data=inputs[:, j], dtype="f8")
        g_out = f.create_group("outputs")
        g_out.create_dataset("psi", data=psi, dtype="f8")
        g_out.create_dataset("success", data=success, dtype=np.bool_)
        g_out.create_dataset("isoflux_used", data=isoflux_used, dtype=np.bool_)
    return path
