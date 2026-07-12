Dataset generation example for fixed-boundary Grad-Shafranov equilibria

This example demonstrates a small, config-driven pipeline that samples tokamak
shape and current parameters, builds LCFS and meshes with autotokamak.core,
solves Grad-Shafranov equilibria (via OpenFUSIONToolkit through autotokamak),
and writes a compact HDF5 dataset ready for surrogate-model training.

Files
- dataset_config.yaml  : example configuration (sampling, parameters, grid,
                         plotting, output path).
- run_dataset_sweep.py  : runner script. Call with the config path as the sole
                         argument.
- README.md             : this file.

Quick start
1) From the workspace root run:
   python run_dataset_sweep.py dataset_config.yaml

2) The script will write outputs/dataset.h5 and create diagnostic plots in
   outputs/plots/ as it runs.

What the runner does
- Samples n_samples parameter vectors (Latin Hypercube or uniform; see
  dataset_config.yaml).
- For each sample:
  - Builds an LCFS with autotokamak.core.geometry.build_lcfs
  - Builds a mesh with autotokamak.core.geometry.build_mesh
  - Calls autotokamak.core.solver.solve_equilibrium with an isoflux
    boundary constraint (the runner records whether the isoflux fitter
    actually held via get_last_solve_info()['isoflux_used']).
  - Interpolates the nodal psi values onto a common (R,Z) grid using
    matplotlib.tri.LinearTriInterpolator (with scipy.griddata fallback).
  - Writes inputs and outputs into outputs/dataset.h5 and updates
    diagnostic plots in outputs/plots/.

Note on isoflux
The installed OpenFUSIONToolkit build in some environments has a known bug
where the isoflux constraint fitter fails. The autotokamak solver wrapper
falls back to an unconstrained solve in that case. Per the example's
requirements, we record get_last_solve_info()['isoflux_used'] per sample but
count unconstrained-fallback solves as successful if the final psi(R,Z)
interpolation contains finite values.

HDF5 layout (outputs/dataset.h5)
- /grid/R, /grid/Z           : 1-D float64 coordinate arrays (len nr, nz)
- /inputs/r0, /a, /kappa, /delta, /Ip : (n_samples,) float64
- /outputs/psi               : (n_samples, nz, nr) float64 (NaN for failed samples)
- /outputs/success           : (n_samples,) bool
- /outputs/isoflux_used      : (n_samples,) bool
- /outputs/error_msgs        : (n_samples,) utf-8 strings
- /outputs/solve_info        : (n_samples,) utf-8 JSON strings (solver info)

Plots produced (outputs/plots/)
- running_success.png  : running success fraction vs sample index
- r0_vs_a.png          : scatter of sampled r0 vs a colored by success
- latest_psi.png       : image of the most recent successful psi(R,Z)
- input_histograms.png : histograms of input parameter distributions

Limitations and notes
- This is a PoC runner for dataset generation. It does not provide
  checkpointing beyond the HDF5 file, nor does it parallelize solves.
- The script relies on the autotokamak.core API; do not modify core/solver.py
  or attempt to change OpenFUSIONToolkit internals to force isoflux success.

Troubleshooting
- If you get ModuleNotFoundError for required packages (numpy, scipy, h5py,
  matplotlib, pyyaml, autotokamak, OpenFUSIONToolkit) the environment is
  misconfigured. Do not pip-install inside this project; report the error.

Contact
- For issues with the isoflux fitter, contact the OpenFUSIONToolkit/maintainer
  team. This runner records the isoflux flag per sample but treats
  unconstrained-fallback solves as valid per the dataset-generation rules.
