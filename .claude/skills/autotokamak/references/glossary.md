# Glossary

Plain-English definitions for autotokamak + fusion-modeling terminology. Adapted from `docs/glossary.md` in the repo.

### Agent
An AI assistant that plans tasks and runs tools or code to complete them. In this repo, URSA's `PlanningAgent` + `ExecutionAgent` pair.

### AutoML
Automatic model search — instead of trying model settings by hand, an optimizer (here: Optuna) searches candidate configs.

### Boundary
The edge of the plasma cross-section. In "fixed-boundary" mode, the user provides the LCFS as an analytic curve; the solver finds ψ inside it.

### Dataset
A collection of solved equilibria used to train/test ML surrogates. Here: an HDF5 file with per-sample inputs `(r0, a, κ, δ, Ip)` and outputs `ψ(R, Z)`.

### Discretization
Turning a smooth PDE into a finite system by putting the domain on a mesh. Knobs: `mesh_dx`, `solver.order`, `boundary.npts`.

### EQDSK
Standard tokamak-equilibrium file format. autotokamak reads legacy EQDSK inputs in `examples/fixed_boundary/`; the modern pipeline uses YAML.

### Equilibrium (MHD equilibrium)
Force balance between plasma pressure and magnetic tension. Grad–Shafranov is the axisymmetric 2D form.

### Fixed-boundary
Solve mode where the LCFS is given up front. This repo's default.

### Free-boundary
Solve mode where the LCFS is determined by external coil currents. Not yet supported here.

### Grad–Shafranov (GS) equation
`Δ*ψ = −μ₀R² p'(ψ) − FF'(ψ)`. The main equation TokaMaker solves.

### HDF5 (`.h5`)
Scientific file format for large arrays + metadata. The dataset generator writes one.

### Hyperparameters
Model settings chosen before training (learning rate, kernel width, PCA components, etc.).

### Interpolation
Estimating values between known points. Used here to place FEM mesh outputs onto a rectangular (R, Z) grid.

### `Ip`
Total plasma current (amperes). One of the five per-sample inputs in the dataset.

### isoflux (constraint)
A boundary condition that forces solved ψ to be constant on the LCFS points. Can fail on extreme shapes → `solve_equilibrium` falls back to unconstrained.

### κ (kappa)
Elongation of the D-shape. 1.0 = circle; 1.6 ≈ ITER; >2 numerically hard.

### δ (delta)
Triangularity of the D-shape. 0.0 = symmetric; positive = inward "D".

### LCFS
Last Closed Flux Surface. The outermost closed ψ contour; the plasma boundary in fixed-boundary mode.

### Mesh
Triangular grid inside the LCFS, built by OFT's `gs_Domain`. Fineness controlled by `mesh_dx`.

### Model zoo
The set of classical ML models the Phase-2 AutoML picks between: GP, kernel ridge, poly-ridge, small MLP.

### OFT (OpenFUSIONToolkit)
The fusion-solver toolkit hosting TokaMaker. Enforces one `OFT_env` per Python kernel — the singleton constraint.

### `OFT_env`
Process-global handle to OFT's Fortran/C runtime. Constructed once per kernel via `get_oft_env()`. See `references/core-api.md`.

### Optimization
Searching parameter space for better scores. Here: Optuna over surrogate hyperparameters.

### Planner / Planning agent
URSA's LLM-driven step generator. Emits an ordered plan; the ExecutionAgent runs each step.

### PoC
Proof of Concept — the current maturity level of the surrogate pipeline.

### Profiles
`p(ψ)`, `F(ψ)`, `q(ψ)` — 1D functions of the flux label. Pressure, poloidal-current function, safety factor.

### `q(ψ)` — safety factor
Toroidal turns per poloidal turn. `q_95` (at 95% flux) is the headline stability metric; physical range ≈ 2–8.

### Solver
TokaMaker: the GS FEM code. Called via `autotokamak.core.solver.solve_equilibrium`.

### Surrogate model
Fast ML approximation to TokaMaker. Trained on the HDF5 dataset; consumed by anything that would otherwise call the full solver.

### TokaMaker
OFT's Grad–Shafranov solver. This repo's "ground truth."

### URSA
LangGraph-based agent framework used to drive plan → execute → feedback loops.

### Workspace
The dir under `examples/` (or `experiments/`) where an agent run writes its generated code, configs, and outputs.

### ψ (psi)
Poloidal flux. The 2D scalar field GS is solved for.
