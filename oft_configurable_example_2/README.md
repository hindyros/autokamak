# OpenFUSIONToolkit (OFT) / TokaMaker — Config-driven Grad–Shafranov equilibrium

This example runs a **TokaMaker Grad–Shafranov equilibrium solve** using **only a single config file** (YAML/JSON) to define:

- the **equation** (currently `grad_shafranov`),
- the **physics targets/parameters** (e.g. `F0`, `Ip`, `Ip_ratio`),
- the **geometry** (analytic LCFS via `create_isoflux`),
- the **discretization** / mesh generation method (e.g. `gs_domain` + `mesh_dx`),
- solver knobs and postprocessing options.

Nothing about the equation/parameters/discretization is hardcoded in the driver script—everything is read from the config.

## Prerequisites

- Linux environment where **OpenFUSIONToolkit** is already built/installed and importable.
- Python with the OFT Python bindings available on `PYTHONPATH`.

This repository/workspace assumes those are already set up (per the task environment). If you are setting up manually:

- Ensure you can run:
  ```bash
  python -c "import OpenFUSIONToolkit as oft; print(oft.__version__ if hasattr(oft,'__version__') else 'import ok')"
  ```
- If you see shared library errors (e.g. `lib...so not found`), you typically need to set `LD_LIBRARY_PATH` to include the OFT build/install lib directory.

## Quickstart

```bash
python run_equilibrium.py --config discretization_config.yaml
```

### Important note about convergence in this environment

In the current OFT build available in this workspace, this analytic setup can fail to converge with errors such as:
- `Matrix solve failed for targets`
- `Isoflux fitting failed`
- `Total poloidal flux is zero`

The driver script is written to be robust anyway:
- it always writes outputs (mesh, resolved config, summary)
- it records pass/fail metrics in `outputs/<case>/summary.yaml`

Once you find a configuration that converges reliably, set strict checks:

```yaml
checks:
  allow_fail: false
  require_finite_psi: true
  targets:
    enabled: true
```

## Running

Run the provided config:

```bash
python run_equilibrium.py --config discretization_config.yaml
```

Run a multi-case discretization sweep (writes derived configs under `./sweeps/`):

```bash
python run_sweep.py --sweep discretization_sweep.yaml
```

Outputs are written under:

```
outputs/<case_name>/
```

where `<case_name>` comes from `case.name` in the config.

## Config overview (key blocks)

The example config is `discretization_config.yaml`.

### `equation`
Selects which equation/workflow to run.

- `equation.type: grad_shafranov`

### `geometry`
Defines the LCFS boundary used to build the GS domain mesh.

- `geometry.lcfs.type: create_isoflux`
- `geometry.lcfs.params` contains the analytic shape parameters passed to OFT `util.create_isoflux`:
  - `R0`, `Z0`: LCFS center (meters)
  - `a`: minor radius (meters)
  - `kappa`: elongation (dimensionless)
  - `delta`: triangularity (dimensionless)
  - `n_pts`: number of points along the boundary polygon

### `mesh`
Controls mesh generation.

- `mesh.method: gs_domain` uses OFT `meshing.gs_Domain`.
- `mesh.gs_domain.mesh_dx` is the characteristic mesh size (meters).

**Important:** this example uses **one plasma region** whose boundary is exactly the LCFS polygon.

### `physics`
Problem parameters and targets.

- `physics.F0`: vacuum toroidal-field function constant used in `TokaMaker.setup(order, F0)`.
- `physics.targets.Ip`: total plasma current target.
- `physics.targets.Ip_ratio`: additional current constraint used by TokaMaker targeting.

Profiles:
- By default the script uses **TokaMaker default profiles** (`profiles.use_defaults: true`).
- Minimal hooks exist in the script for custom profiles, but the provided example does **not** require them.

### `solver`
Discretization order and optional solver settings.

- `solver.order`: polynomial order passed to `TokaMaker.setup(order, F0)`.
- `solver.settings_patch`: optional dictionary of attributes to apply onto `mygs.settings` before `update_settings()`.

### `postprocess`
Controls optional output/plot generation.

- Mesh is always saved.
- If ψ values are available, the script attempts to write them and plot contours.

### `checks`
Quantitative “definition of done” checks recorded in `summary.yaml`.

- `checks.allow_fail`: if `false`, the script exits nonzero when checks fail.
- Checks include:
  - `solve_ok`: whether `mygs.solve()` returned without exception.
  - `psi_hat_finite`: whether normalized ψ on nodes exists and is finite.
  - `targets.Ip`: best-effort comparison of requested vs achieved `Ip` if an achieved value is available in `get_stats()`.

## Units and conventions

This example assumes typical GS/TokaMaker conventions:

- Lengths (`R0`, `Z0`, `a`, `mesh_dx`) are in **meters**.
- `Ip` is in **Amperes**.
- `F0` units follow the Grad–Shafranov convention (commonly T·m); use values consistent with your OFT/TokaMaker setup.

The script does not re-normalize these; it passes them to OFT as provided.

## Outputs

For a run with `case.name: demo_isoflux` you should find:

```
outputs/demo_isoflux/
  resolved_config.yaml      # the fully resolved config used for the run
  mesh.h5                   # mesh saved via OFT (gs_Domain output)
  psi_nodes.npy             # ψ values on mesh nodes (only if available)
  summary.yaml              # run summary, solver status, and checks
  mesh.png                  # mesh plot (if plotting succeeds)
  psi_contours.png          # ψ contour plot (if ψ is available and finite)
```

Notes:
- If the solve fails early, `psi_nodes.npy` and `psi_contours.png` may be absent.
- `summary.yaml` is always written and records exceptions and check results.

## Troubleshooting

### Import errors (`ModuleNotFoundError: OpenFUSIONToolkit`)
- Ensure OFT’s Python bindings are on `PYTHONPATH`.
- In some setups, you may need to `source` an environment script provided by your OFT build.

### Shared library errors (`lib*.so: cannot open shared object file`)
- Add the OFT library directory to `LD_LIBRARY_PATH`.

### Mesh generation failures
- Decrease `mesh.gs_domain.mesh_dx` (finer mesh) or increase it (coarser mesh) depending on geometry.
- Ensure the LCFS is not self-intersecting; try higher `n_pts`.

### Non-convergence / `Matrix solve failed for targets`
This indicates TokaMaker’s target system could not be satisfied with the current mesh/order/targets.

Try one or more of:
- Change targets: adjust `physics.targets.Ip` and/or `physics.targets.Ip_ratio`.
- Reduce polynomial order: set `solver.order: 1`.
- Change mesh resolution: adjust `mesh_dx`.
- Add/adjust solver settings in `solver.settings_patch` (exact options depend on OFT version).

To force the run to **fail fast** in scripts/CI, set:

```yaml
checks:
  allow_fail: false
```

Then the script will exit with a nonzero code if the checks fail.
