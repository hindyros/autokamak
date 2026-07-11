# YAML config reference

autotokamak has three physics-config shapes plus one agent-config shape. Prompt YAMLs (the agent-config shape) live under `src/autotokamak/agent/prompts/`; physics YAMLs live under `examples/**/`.

Rule of thumb: if it starts with `problem:`, it's an agent config. If it starts with `equation:` or `sampling:` or `base_config:`, it's a physics config.

## 1. Equilibrium config (single solve)

Consumed by: `examples/config_driven_equilibrium/run_equilibrium_from_config.py`, `EquilibriumConfig.from_yaml`.
Starter: `assets/templates/equilibrium.yaml`.

```yaml
equation:
  name: gs                    # only "gs" supported

boundary:
  type: isoflux               # only "isoflux" (analytic D-shape) supported
  npts: 80                    # 0 < npts <= 1000; go below 40 and add_polygon self-intersects
  r0: 0.42                    # major radius of LCFS center (m)
  z0: 0.0                     # vertical center of LCFS (m)
  a:  0.15                    # minor radius (m)
  kappa: 1.4                  # elongation, 0.5..3.0 (1.0 = circle)
  delta: 0.0                  # triangularity, -1.0..1.0

mesh:
  method: gs_domain           # only gs_domain
  regions:
    - name: plasma
      type: plasma
      dx: 0.015               # target triangle edge length (m). <0.010 gets expensive fast.

solver:
  order: 2                    # FE polynomial order, 1..3
  F0: 0.10752                 # toroidal-field constant F0 (T·m)
  full_domain: false          # false = solve half-plane and reflect
  maxits: 60                  # nonlinear iteration cap
  free_boundary: false        # true = solve for LCFS instead of fixing it

targets:                      # AT LEAST ONE of these keys must be present
  Ip: 120000.0                # total plasma current (A)
  Ip_ratio: 1.0

init_psi:
  method: tokamaker_default   # or "isoflux" (shape-aware, but can fail on extreme κ/δ)

outputs:
  out_dir: outputs            # relative to CWD; use unified_output_dir for standard layout
  mesh_png: mesh.png
  psi_png: psi.png

meta:                         # ignored by loader; free-form provenance
  oft_version_min: "1.0.0"
```

**Gotchas:**
- `init_psi.method: isoflux` triggers a shape-aware seed. On extreme shapes OFT's internal isoflux fit throws; `solve_equilibrium` catches and downgrades to `init_psi(-1.0)` (uniform-current seed) with a warning. If you see the warning, either accept it or fall back to `tokamaker_default`.
- `targets` is the most common validation failure — forgetting to set even one key rejects the whole config.
- `outputs.out_dir` is CWD-relative unless absolute. For repeatable layouts wrap the runner in `autotokamak.core.io.unified_output_dir`.

## 2. Sweep config (many solves)

Consumed by: `examples/config_driven_equilibrium/run_discretization_sweep.py`, `SweepConfig.from_yaml`.
Starter: `assets/templates/sweep.yaml`.

```yaml
base_config: discretization_config.yaml   # relative to this file

cases:
  - case_id: dx0.020_p1
    overrides:
      mesh:
        regions:
          - name: plasma
            dx: 0.020
      solver:
        order: 1

  - case_id: dx0.010_p2
    overrides:
      mesh:
        regions:
          - name: plasma
            dx: 0.010
      solver:
        order: 2
```

`overrides:` is deep-merged into the loaded base config; the merged dict is then validated by `EquilibriumConfig`. Only the deltas need to appear.

**Gotchas:**
- `mesh.regions` is a list — override the whole list, not a single entry by index.
- `case_id` must be filesystem-safe (used as a directory name); the runner sanitizes non-alphanumerics.

## 3. Dataset generation config (Phase-1 sweep)

Consumed by: `examples/dataset_generation/run_dataset_sweep.py`.
Starter: `assets/templates/dataset_config.yaml`.

```yaml
sampling:
  method: lhs                 # currently only "lhs"
  n_samples: 500
  seed: 0

parameters:                   # bounds for the 5 free knobs; LHS-sampled inside these
  r0:    {low: 0.35, high: 0.55}
  a:     {low: 0.10, high: 0.20}
  kappa: {low: 1.0,  high: 1.6}
  delta: {low: 0.0,  high: 0.4}
  Ip:    {low: 80000, high: 200000}

fixed:                        # applied to every sample
  z0: 0.0
  F0: 0.10752
  npts: 80
  mesh_dx: 0.015
  solver_order: 1
  Ip_ratio: 1.0

output_grid:                  # (R, Z) grid ψ is interpolated onto
  R: {min: 0.15, max: 0.80, n: 64}
  Z: {min: -0.40, max: 0.40, n: 96}

output_path: "outputs/dataset.h5"

plotting:                     # per-sample diagnostic plots (expensive for large N)
  enabled: true
  every_n_samples: 1
  output_dir: "outputs/plots"
```

**Gotchas:**
- `parameters` bounds outside the feasible box (see `references/physics.md`) will produce many isoflux fallbacks. Run `scripts/probe_feasible.py` first to pick bounds.
- The output grid dimensionality drives PCA cost downstream — 64×96 = 6144 dims is the shipped baseline. Doubling to 128×192 quadruples training time.
- HDF5 output is written incrementally; a crashed sweep leaves a partial `.h5`. Delete and re-run.

## 4. Agent prompt config

Consumed by: `plan_execute.py`, `plan_execute_feedback.py`, `meta_loop.py`.

Common shape (all keys are top-level):

```yaml
problem: |                    # required — the natural-language task
  Build a fixed-boundary GS parameter sweep that writes a surrogate-training
  dataset to <workspace>/outputs/dataset.h5 ...

workspace: examples/dataset_generation
model: openai:gpt-5-mini      # LLM string for langchain init_chat_model
symlinks:                     # symlinked into workspace so agent can read read-only source
  - src: ./ursa
    dst: ursa
  - src: ./OpenFUSIONToolkit
    dst: OpenFUSIONToolkit

feedback_rounds: 2            # plan_execute_feedback only
validate_after: true          # plan_execute_feedback only

scorer: metric_surrogate.score_surrogate_run  # dotted path to a scorer callable
scorer_kwargs: {}
expected_artifacts:
  - outputs/dataset.h5
  - outputs/summary.json

allowed_root_files:           # workspace hygiene whitelist (plan_execute_feedback)
  - run.py
  - config.yaml
  - README.md
infra_root_files:
  - ursa
  - OpenFUSIONToolkit
```

The **`meta_loop`** prompt shape is different — it forces structured output and does not use `problem:`. See `assets/templates/meta_loop_config.yaml` or `references/agent-runners.md#meta-loop`.

**Constraints blocks:** every prompt's `problem:` includes a hard `CONSTRAINTS:` block forbidding `git`, `pip install`, and `input()` calls. Preserve these when editing.

**Where the shipped prompts live:**

| Prompt YAML | Workspace |
|---|---|
| `dataset_generation.yaml` | `examples/dataset_generation/` |
| `oft_example_generation.yaml` | `examples/fixed_boundary/` |
| `oft_discretization_example.yaml` | `examples/config_driven_equilibrium/` |
| `surrogate_automl.yaml` | `examples/surrogate_automl/` |
| `surrogate_meta.yaml` | `examples/surrogate_meta/` (routed to `meta_loop`) |
