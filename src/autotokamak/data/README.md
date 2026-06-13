# `autotokamak.data`

Sweep generators and dataset loaders for training surrogate models.

Populated in **Week 2** of the research plan. Planned modules:

| Module | Purpose |
|---|---|
| `sweep.py` | Programmatic parameter-grid generator that fans out TokaMaker runs |
| `loader.py` | PyTorch `Dataset` over the consolidated HDF5 dataset (`data/processed/*.h5`) |
| `interpolate.py` | Resample each equilibrium's $\psi$ onto a common $(R, Z)$ grid |

Raw outputs go to `data/raw/` (gitignored); consolidated training files go to `data/processed/`.
