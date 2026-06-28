"""Error metrics for ψ(R,Z) predictions.

All metrics operate on full-grid arrays of shape ``(N, nz, nr)`` after any
PCA inverse-transform. They share a NaN-handling convention: if both true and
pred have NaN at the same cell, that cell is excluded; if only one has NaN,
the cell is excluded and counted in a returned ``n_excluded`` field when the
function returns a diagnostic dict (the scalar variants just exclude).

This module is intentionally tiny — the agent's runner should call these
rather than re-implementing them. The DSPy scorer also imports them so the
score uses the same definitions as the agent's reported numbers.
"""

from __future__ import annotations

import numpy as np


def _valid_mask(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Return a boolean array of cells to include in the metric.

    Excludes any cell where either array is non-finite. If ``mask`` is given,
    its False entries are also excluded (i.e. the metric is computed only
    over cells where the mask is True AND both arrays are finite).
    """
    valid = np.isfinite(true) & np.isfinite(pred)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return valid


def psi_rmse(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Cell-averaged RMSE of ψ over included cells.

    Computed in physical ψ units (whatever the dataset HDF5 stored). For
    surrogate scoring we usually want this in the ORIGINAL ψ space, not in
    PCA-coefficient space — call ``inverse_transform`` first.
    """
    valid = _valid_mask(true, pred, mask)
    if not valid.any():
        return float("nan")
    err = np.asarray(true, dtype=np.float64)[valid] - np.asarray(pred, dtype=np.float64)[valid]
    return float(np.sqrt(np.mean(err * err)))


def relative_l2(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """||true - pred||_2 / ||true||_2 over included cells.

    Scale-invariant — useful when comparing across samples with different
    overall ψ magnitudes. Returns NaN if ``true`` is identically zero on
    included cells.
    """
    valid = _valid_mask(true, pred, mask)
    if not valid.any():
        return float("nan")
    t = np.asarray(true, dtype=np.float64)[valid]
    p = np.asarray(pred, dtype=np.float64)[valid]
    denom = float(np.linalg.norm(t))
    if denom < 1e-30:
        return float("nan")
    return float(np.linalg.norm(t - p) / denom)


def pixelwise_max_err(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Max absolute error over included cells. Useful for sanity-checking outliers."""
    valid = _valid_mask(true, pred, mask)
    if not valid.any():
        return float("nan")
    err = np.abs(
        np.asarray(true, dtype=np.float64)[valid]
        - np.asarray(pred, dtype=np.float64)[valid]
    )
    return float(np.max(err))


def baseline_mean_predictor_rmse(psi_train: np.ndarray, psi_val: np.ndarray) -> float:
    """RMSE of the trivial cell-wise mean predictor.

    Predict val ψ as the per-pixel mean of train ψ. This is the "did our
    surrogate beat doing nothing" reference point and is the denominator in
    the scorer's ``val_rmse_vs_baseline`` quality term.
    """
    if psi_train.ndim != 3 or psi_val.ndim != 3:
        raise ValueError("psi_train and psi_val must be shape (N, nz, nr)")
    # Outside-LCFS pixels are all-NaN columns; silence the resulting warning.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_psi = np.nanmean(psi_train, axis=0)  # (nz, nr)
    # Broadcast mean over the val N axis.
    pred = np.broadcast_to(mean_psi[None, :, :], psi_val.shape)
    return psi_rmse(psi_val, pred)


__all__ = [
    "baseline_mean_predictor_rmse",
    "pixelwise_max_err",
    "psi_rmse",
    "relative_l2",
]
