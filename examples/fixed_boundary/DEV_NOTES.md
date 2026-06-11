# DEV_NOTES (OpenFusionToolkit TokaMaker fixed-boundary example)

This workspace contains a standalone, runnable Python translation of the upstream notebook:
`OpenFusionToolkit/src/examples/TokaMaker/fixed_boundary/fixed_boundary_ex1.ipynb`.

## Intentional deviations / robustness tweaks

1. **EQDSK file path resolution**
   - The notebook calls `read_eqdsk('gNT_example')` assuming the working directory is the notebook directory.
   - In `run_fixed_boundary_equilibrium.py`, we resolve the EQDSK file explicitly at:
     `../OpenFusionToolkit/src/examples/TokaMaker/fixed_boundary/gNT_example` (relative to this script),
     and fall back to `./gNT_example` if that path does not exist.

2. **EQDSK mesh resolution default**
   - The notebook sets `mesh_dx = 0.15` for the EQDSK case.
   - The script now uses `0.15` as the default *unless* the user overrides via `--mesh-dx`.

3. **EQDSK profile application fallback**
   - The notebook builds normalized `linterp` profiles for `ffprim` and `pprime` and applies them via `mygs.set_profiles()`.
   - On this system/build, applying these custom profiles caused `mygs.solve()` to fail with:
     `Matrix solve failed for targets`.
   - The script therefore retries the EQDSK solve once **without** calling `set_profiles()` if the first attempt fails.
   - The summary written to `summary.json` records this deviation in `deviations_from_notebook`.

## Notes

- The analytic case follows the notebook closely and runs without special fallbacks.
- Plotting is optional and disabled with `--no-plots` (useful for headless environments).
