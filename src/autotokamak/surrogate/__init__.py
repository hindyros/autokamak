# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Structured Phase-2 surrogate AutoML: sklearn model zoo + Optuna search.

Submodules:
- :mod:`autotokamak.surrogate.dataset`       — HDF5 dataset loading + k-fold splits
- :mod:`autotokamak.surrogate.reduce`        — PCA reduction of psi maps
- :mod:`autotokamak.surrogate.metrics`       — psi-RMSE and baseline metrics
- :mod:`autotokamak.surrogate.zoo`           — sklearn model factories (``make_model``)
- :mod:`autotokamak.surrogate.schema`        — ``SurrogateConfig``/``SearchSpec``/``StudyResult``
- :mod:`autotokamak.surrogate.optuna_search` — Optuna study harness (inner loop)
- :mod:`autotokamak.surrogate.automl_loop`   — LLM-driven outer search loop (Phase-2)
- :mod:`autotokamak.surrogate.diagnostics`   — bottleneck diagnostics for the meta-agent

Starting points:

>>> from autotokamak.surrogate import load_dataset, kfold, make_model
>>> bundle = load_dataset("dataset.h5")        # -> DatasetBundle(inputs, psi, R, Z, ...)
>>> model = make_model("kernel_ridge", alpha=1e-3)

``run_automl_loop`` is the outer search; it takes a ``decision_fn`` so the same
loop runs under a scripted policy (L0) or a typed LLM picker (L1). ``PCAModel``
in :mod:`~autotokamak.surrogate.reduce` is the reduction the whole pipeline
hangs on: the flux field is low-rank, but only if you keep it in physical
webers rather than normalising per sample.
"""

from .automl_loop import run_automl_loop
from .dataset import DatasetBundle, Splits, kfold, load_dataset
from .reduce import PCAModel
from .zoo import make_model

__all__ = [
    "DatasetBundle",
    "PCAModel",
    "Splits",
    "kfold",
    "load_dataset",
    "make_model",
    "run_automl_loop",
]
