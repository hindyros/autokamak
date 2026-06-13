# `autotokamak.eval`

Evaluation harness comparing surrogate predictions against TokaMaker ground truth.

Populated in **Week 3** (baseline harness) and extended through **Week 6**. Planned modules:

| Module | Purpose |
|---|---|
| `metrics.py` | Pointwise $\psi$ RMSE, magnetic-axis error, $q_{95}$ error, latency |
| `benchmark.py` | Run surrogate + OFT on the same input grid, produce comparison tables |
| `plots.py` | Side-by-side $\psi$ contour plots, error heatmaps, q-profile overlays |
