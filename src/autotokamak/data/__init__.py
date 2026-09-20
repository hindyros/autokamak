# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Data loaders and sweep generators for equilibrium training datasets.

>>> from autotokamak.data import SweepConfig, run_sweep, read_h5_arrays
>>> arrays = read_h5_arrays("dataset.h5")     # -> DatasetArrays(R, Z, inputs, psi, ...)

``run_sweep`` drives the solver over either a config-described grid or an
explicit ``(N, 5)`` matrix of parameter vectors, and writes HDF5 in the layout
the rest of the package expects. ``PARAM_ORDER`` fixes the column order of
that matrix and is worth importing rather than retyping.
"""

from .h5io import DatasetArrays, read_h5_arrays, write_h5_arrays
from .schema import PARAM_ORDER, SweepConfig
from .sweep import run_sweep

__all__ = [
    "PARAM_ORDER",
    "DatasetArrays",
    "SweepConfig",
    "read_h5_arrays",
    "run_sweep",
    "write_h5_arrays",
]
