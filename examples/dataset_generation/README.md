# Fixed-boundary Grad–Shafranov dataset generation (autotokamak)

This example generates a small dataset of **fixed-boundary** Grad–Shafranov (GS)
equilibria using `autotokamak.core`. It sweeps a few geometry / physics parameters
(`r0`, `a`, `kappa`, `delta`, `Ip`), solves one equilibrium per sample, and interpolates
poloidal flux `psi(R,Z)` onto a common rectilinear grid for surrogate-model training.

## Requirements / preflight

This example expects a Python 3.11 environment with these importable packages:
`PyYAML (yaml)`, `h5py`, `numpy`, `scipy`, `matplotlib`, and `autotokamak`
(plus its solver backend via OpenFUSIONToolkit/TokaMaker).

The runner performs a lightweight **preflight** at startup: it checks the Python
version and that these imports succeed before attempting the sweep.

## Run

From this directory:

```bash
python run_dataset_sweep.py dataset_config.yaml
```

Output is written to `output.dataset_path` (default: `outputs/dataset.h5`).

## YAML config schema (what matters)

See `dataset_config.yaml` for a concrete config. Key fields:

- `sampling.method`: `lhs` (Latin hypercube) or `uniform`
- `sampling.n_samples`, `sampling.seed`
- `parameters`: mapping of swept parameters to `{low, high}` (order matters)
- `fixed`: `z0`, `F0`, `npts`, `mesh_dx`, `solver_order` (must be 1), `Ip_ratio`
- `output_grid`: `r: {min, max, n}` and `z: {min, max, n}`
- `output.dataset_path`: HDF5 output path

## Conventions and failure semantics

- Coordinates: `R` is major radius [m], `Z` is vertical [m].
- `psi` is the Grad–Shafranov poloidal flux as returned by TokaMaker and then
  interpolated to the `(R,Z)` grid. Units/normalization are solver-defined.
- `success[i] == True` means the equilibrium solve completed without raising and the
  grid interpolation produced an array for sample `i`.
- If a sample fails (solve or interpolation exception), its `psi[i, :, :]` is left as
  `NaN` everywhere and `success[i] == False`.
- Even for successful samples, points outside the triangulation / convex hull may
  remain `NaN` after interpolation.

## Dataset layout (HDF5)

All datasets use gzip compression (level 4). `/outputs/psi` is chunked as `(1, nz, nr)`
for efficient per-sample reads.

- `/grid/R` : `(nr,)` float64, R coordinates [m]
- `/grid/Z` : `(nz,)` float64, Z coordinates [m]

- `/inputs/r0`    : `(N,)` float64 [m]
- `/inputs/a`     : `(N,)` float64 [m]
- `/inputs/kappa` : `(N,)` float64 [1]
- `/inputs/delta` : `(N,)` float64 [1]
- `/inputs/Ip`    : `(N,)` float64 [A]

- `/outputs/psi`     : `(N, nz, nr)` float64, axis order `psi[sample, z_index, r_index]`
- `/outputs/success` : `(N,)` bool

Self-describing/provenance extras:
- `/config_yaml` : scalar bytes dataset with the full YAML config text
- `/parameter_names` : `(5,)` list of swept parameter names (stored order)
- `/parameter_bounds` : `(5,2)` float64 array of `[low, high]` (columns annotated)
- `/inputs_matrix` : `(N,5)` float64 matrix matching `/parameter_names`
- root attributes include: `created_utc`, `git_commit` (if available), units, and
  interpolation method notes
