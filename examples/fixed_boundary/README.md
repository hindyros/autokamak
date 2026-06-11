# OpenFUSIONToolkit TokaMaker — Fixed-Boundary Equilibrium (Standalone Script)

This workspace provides a **standalone** Python script that builds and solves a **fixed-boundary Grad–Shafranov equilibrium** using **OpenFUSIONToolkit (OFT) / TokaMaker**.

Reference (read-only, not modified):
- `OpenFusionToolkit/src/examples/TokaMaker/fixed_boundary/fixed_boundary_ex1.ipynb`

What you run here:
- `run_fixed_boundary_equilibrium.py` (writes outputs to `./outputs/...`; it does **not** write into `./OpenFusionToolkit/` or `./ursa/`).

## What the script does

For either of two cases:
1. **Create/read the LCFS boundary** (fixed plasma boundary)
2. **Build a GS domain mesh** for that boundary
3. Configure TokaMaker in **fixed-boundary mode** (`free_boundary = False`)
4. Apply **targets** (e.g., total plasma current) and (optionally) **profiles**
5. **Solve** the equilibrium
6. **Postprocess** and save artifacts (NPZ/JSON) and optional headless plots (PNG)

Supported cases:
- `--case analytic`: boundary is generated analytically (isoflux surface / shaped LCFS)
- `--case eqdsk`: boundary is loaded from OFT’s bundled EQDSK example (`gNT_example`)

## Prerequisites / environment

This assumes the execution environment already provides:
- Python 3
- `OpenFUSIONToolkit` importable from Python (already on `PYTHONPATH`)
- OFT binaries available on `PATH` (not strictly required for this script, but typically present)

No package installation steps are required or performed by this example.

## How to run (exact commands)

Run the **analytic** fixed-boundary equilibrium:

```bash
python run_fixed_boundary_equilibrium.py --case analytic
```

Run the **EQDSK**-based fixed-boundary equilibrium:

```bash
python run_fixed_boundary_equilibrium.py --case eqdsk
```

Write results to a specific directory (a timestamped subdirectory is created inside):

```bash
python run_fixed_boundary_equilibrium.py --case analytic --outdir ./outputs/my_run
```

Disable plotting (useful on headless systems; the run still saves `.npz`/`.json`):

```bash
python run_fixed_boundary_equilibrium.py --case analytic --no-plots
```

## CLI options

Commonly used options (see `--help` for the full list):

- `--case {analytic,eqdsk}`: which boundary source to use
- `--outdir PATH`: top-level output directory
- `--no-plots`: skip saving PNG plots

Debug / workflow helpers:
- `--imports-only`: stop after validating imports
- `--setup-only`: stop after building boundary + mesh + configuring the solver
- `--smoke-solve`: attempt a minimal solve (faster, for quick validation)

## Where outputs appear

Each run creates a directory like:

- `outputs/YYYYMMDDTHHMMSSZ/`

Inside that directory:

- `run.log` — complete mirrored console log (the script is intentionally verbose)
- `summary.json` — key settings and summary values
- `lcfs_contour.npz` — LCFS contour arrays
- `mesh.npz` — GS domain mesh arrays
- `profiles.npz` — returned profiles (pressure/current/etc.)
- `q_and_geometry.npz` — q-profile and derived geometry arrays
- `mesh.png`, `psi.png` (and for EQDSK: `eqdsk_comparison.png`) — plots (unless `--no-plots`)

## Troubleshooting

1. **Start with the log**: inspect `outputs/.../run.log`.

2. **Import errors (`ModuleNotFoundError: OpenFUSIONToolkit ...`)**
   - This indicates OFT is not on `PYTHONPATH` in your shell/session.
   - In this task environment it should already be configured.

3. **Headless plotting issues**
   - The script uses Matplotlib’s non-interactive backend (`Agg`) by default.
   - If you still want to skip plot creation entirely, use `--no-plots`.

4. **EQDSK solve robustness**
   - Some EQDSK/profile combinations can be numerically finicky.
   - The script includes a one-time fallback retry in the EQDSK case: if the solve fails when custom profiles are applied, it retries with default profiles to ensure the example completes. This is intentional and documented.
