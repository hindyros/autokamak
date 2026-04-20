# OpenFUSIONToolkit (OFT) TokaMaker equilibrium — config-driven discretization example

This workspace contains a **single, config-driven example** of running a Grad–Shafranov equilibrium with **OpenFUSIONToolkit / TokaMaker**.

Key properties of the example:
- **All physics + discretization choices come from a YAML config** (no hardcoded `mesh_dx`, polynomial order, targets, etc.).
- Builds an **analytic LCFS** via `create_isoflux(R0, Z0, a, kappa, delta, n_pts)`.
- Builds a 2D triangular mesh via `gs_Domain` and plots it via `gs_mesh.plot_mesh(fig, ax)`.
- Solves and plots flux surfaces via `TokaMaker.plot_psi(fig, ax)`.
- Writes per-run outputs under `outputs/` including a copy of the exact resolved config (`config_used.yaml`) for reproducibility.

## Prerequisites

- Python: use the Python interpreter in the environment where OFT is already available.
- OpenFUSIONToolkit: must already be installed and on `PYTHONPATH`.
  - This environment reports (from OFT startup banner):
    - Development branch: `gh-pages`
    - Revision id: `8905cc5`
    - Not compiled with MPI

No additional packages beyond those already in the environment are required.

## Files (minimal deliverables)

- `discretization_config.yaml` — example config specifying equation, parameters, targets, and discretization.
- `run_equilibrium_from_config.py` — single-case runner (loads a config and runs one solve).
- `run_invert_psi.py` — optional outer loop: tune YAML parameters so `get_psi()` matches a target ψ (see `README_INVERT_PSI.md`). Uses `forward_once.py` subprocesses so OFT is not re-initialized in-process between solves; `forward_plot_psi.py` writes stage **ψ** PNGs after a run.
- `invert_psi_example.yaml` — small inversion demo (`F0` + `Ip`, `dof` ψ loss + regularization); see `README_INVERT_PSI.md`.
- `run_discretization_sweep.py` — sweep runner (runs multiple cases by overriding discretization fields).
- `sweep_discretizations.yaml` — example sweep definition.
- `run_new_discretizations.yaml` — template/instructions for a downstream agent to generate new configs.
- `outputs/` — generated after running.

## Run (single case)

```bash
python run_equilibrium_from_config.py discretization_config.yaml
```

Expected console output includes mesh statistics, the OFT initialization banner, and solver iteration lines.

Expected output tree (example):

```text
outputs/
  dx0.015_p2_n80_<hash>/
    config_used.yaml
    summary.json
    summary.yaml              # may be omitted or replaced with a pointer if YAML serialization fails
    raw_arrays.npz
    mesh_dx0.015_p2_n80.png
    psi_dx0.015_p2_n80.png
```

## Run (discretization sweep)

A sweep file lets you keep the physics fixed while varying discretization (e.g. mesh size and FE order).

```bash
python run_discretization_sweep.py sweep_discretizations.yaml
```

The sweep runner:
- Loads `base_config`.
- Applies each case’s `overrides` (deep-merge; `mesh.regions` overrides matched by `name`).
- Writes temporary per-case configs under `outputs/_sweep_tmp_configs/`.
- Runs `run_equilibrium_from_config.py` once per case.

## Config overview (important fields)

All configuration is in YAML. The runner expects the following *conceptual* structure (see `discretization_config.yaml` for an exact example).

### Equation / model
- `equation`: currently intended for Grad–Shafranov / TokaMaker runs.

### Analytic boundary (LCFS)
- `boundary.type: isoflux`
- `boundary.isoflux`:
  - `R0`, `Z0`: geometric center (m)
  - `a`: minor radius (m)
  - `kappa`: elongation
  - `delta`: triangularity
  - `n_pts`: number of points used to discretize the contour

### Discretization (mesh + FE space)
- `mesh.regions`: list of regions; each region has:
  - `name`: used for override matching in sweep
  - `dx`: target mesh spacing (e.g. `0.015`)
  - `tag`: region label (passed through to OFT region definition)
- `solver.order`: polynomial order for the FE space (typical values: `1`, `2`, ...)

### Targets and solver tolerances
- `targets`: targets passed to `TokaMaker.set_targets(...)` (example: `Ip`, `Ip_ratio`, etc.).
- `solver`: numerical settings (typical entries in this example: `order`, `F0`, nonlinear iteration limits / tolerances).

## Reproducibility / provenance

Each run writes:
- `config_used.yaml`: exact config used for that case (after any sweep overrides).
- `summary.json`: mesh statistics + selected equilibrium scalars returned by OFT.
- Case directory name includes both a human-readable discretization slug and a hash of the config content.

## Troubleshooting

### Import errors (cannot import OpenFUSIONToolkit)
- Ensure you are using the same Python environment where OFT is installed.
- Check that `PYTHONPATH` includes the OFT Python package directory.

### Mesh generation errors
- If meshing fails, try:
  - increasing `boundary.isoflux.n_pts` (smoother boundary polygon), or
  - using a slightly larger `mesh.regions[].dx`.

### Isoflux constraint failures
Some OFT/TokaMaker builds can fail during the internal isoflux-fitting step on certain meshes/shapes:

- You may see: `Isoflux fitting failed`
- This example is designed to **still complete**: it tries the isoflux-constrained solve first and, on failure, re-runs an **unconstrained** solve (and prints a warning).

### Non-convergence
- Try reducing the nonlinearity (e.g. lower current targets) or relaxing solver tolerances / increasing iteration limits in `solver`.

### Output / permission issues
- Ensure the current directory is writable (the code writes under `outputs/`).
