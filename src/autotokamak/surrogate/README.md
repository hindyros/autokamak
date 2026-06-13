# `autotokamak.surrogate`

Neural surrogate models for the Grad-Shafranov equation.

Populated in **Week 4** of the research plan. Planned modules:

| Module | Purpose |
|---|---|
| `baseline_nn.py` | Nearest-neighbor interpolation in $(R_0, a, \kappa, \delta, I_p)$ space |
| `baseline_mlp.py` | Flat MLP from 5-d input to $H \times W$ flattened $\psi$ |
| `deeponet.py` | DeepONet (branch over physics, trunk over $(R, Z)$) |
| `fno.py` | Fourier Neural Operator over the $(R, Z)$ grid |

Trained checkpoints go to `models/checkpoints/` at repo root, not into this package.
