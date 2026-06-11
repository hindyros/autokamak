# Fixed-boundary TokaMaker example: standardized output artifacts

This document defines a stable, reproducible set of output artifacts that the standalone script (to be created in later steps) will generate.

## Output directory layout

All artifacts will be written under a single run directory:

```
outputs/fixed_boundary_ex1/<run_id>/
```

Where `<run_id>` is reproducible by default and can also be overridden:

- Default: a timestamp in UTC: `YYYYmmdd_HHMMSSZ`
- Optional: user-provided string via `--run-id RUN_ID`

### Required artifacts (always generated)

1. **Run log (text)**
   - Path: `outputs/fixed_boundary_ex1/<run_id>/run.log`
   - Format: plain text
   - Contents:
     - echoed key configuration parameters
     - major progress milestones (mesh creation, solver init, solve, exports)
     - any caught exceptions with stack trace

2. **Key scalar summary (JSON)**
   - Path: `outputs/fixed_boundary_ex1/<run_id>/summary.json`
   - Format: JSON (UTF-8)
   - Contents (minimum):
     - `run_id`
     - `created_utc`
     - `oft_version` (if available from package)
     - `inputs`: mesh settings, target parameters
     - `results`: key equilibrium scalars available from TokaMaker after solve (e.g., Ip target/achieved if available)
     - `files`: relative paths to generated files

3. **Equilibrium definition / solution export**
   The notebook does not explicitly export an EQDSK file, but it *does* read an EQDSK and solve a fixed-boundary equilibrium.
   For a stable artifact, the script will export:

   - **Computed profiles (NumPy)**
     - Path: `outputs/fixed_boundary_ex1/<run_id>/profiles.npz`
     - Format: NumPy `.npz`
     - Arrays (as available): `psi`, `f`, `fp`, `p`, `pp`

   - **Safety factor and geometry summary (NumPy)**
     - Path: `outputs/fixed_boundary_ex1/<run_id>/q_and_geometry.npz`
     - Format: NumPy `.npz`
     - Arrays (as available): `psi_q`, `qvals`, `ravgs`, `dl`, `rbounds`, `zbounds`

   - **Input LCFS contour (NumPy)**
     - Path: `outputs/fixed_boundary_ex1/<run_id>/lcfs_contour.npz`
     - Format: NumPy `.npz`
     - Arrays: `R`, `Z` (from `LCFS_contour`)

   Notes:
   - If OpenFUSIONToolkit exposes an EQDSK writer in Python, we will additionally write:
     - `equilibrium.eqdsk`
     - but this is **optional** and will be best-effort.

4. **Mesh dump (NumPy)**
   - Path: `outputs/fixed_boundary_ex1/<run_id>/mesh.npz`
   - Format: NumPy `.npz`
   - Arrays: `mesh_pts`, `mesh_lc`
   - Plus `mesh_reg` if returned and serializable.

### Optional artifacts (generated unless `--no-plots`)

All plots must be headless-safe (matplotlib Agg backend) and saved as PNG.

- `plots/mesh.png` — mesh plot
- `plots/psi.png` — flux surface plot
- `plots/profiles_comparison.png` — FF', P', q comparison plot
- `plots/profiles_final.png` — final FF', P', q, <Jphi> plot

Paths:

```
outputs/fixed_boundary_ex1/<run_id>/plots/*.png
```

### Optional artifacts (debug)

- `inputs/` copy of any small, local input files used (e.g. `gNT_example`) if we choose to copy it for provenance.
  - This will be controlled by `--copy-inputs`.

## CLI flags (to support reproducibility)

The future script will expose:

- `--outdir PATH` (default: `outputs/fixed_boundary_ex1`)
- `--run-id RUN_ID` (default: timestamp)
- `--no-plots` (disable PNG creation)
- `--nthreads N` (default: 2)
- `--mesh-dx FLOAT` (default: notebook default for the chosen case)

## Reproducibility considerations

- The solver is deterministic given the same mesh and settings.
- The script will print and record all key settings in `run.log` and `summary.json`.
- All arrays exported in `.npz` provide a stable, version-independent data interchange format.
