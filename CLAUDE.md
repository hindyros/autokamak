# CLAUDE.md — Repo guide for `autotokamak`

## What this repo is

The **`autotokamak`** package — an evolving platform for two complementary research threads:

1. **ML surrogate models** for the Grad–Shafranov equation (fast approximations of TokaMaker's FEM solver).
2. **LLM-driven agentic workflows** (via [URSA](https://github.com/lanl/ursa)) that plan and run equilibrium computations end-to-end.

It builds on:

- **URSA** — LangChain/LangGraph-based `PlanningAgent` + `ExecutionAgent` pair.
- **OpenFUSIONToolkit (OFT) / TokaMaker** — the ground-truth Grad–Shafranov solver. Installed via `pip install OpenFUSIONToolkit>=26.6`.

## Top-level layout

```
agentic_simulations/                # repo root (will be renamed to autotokamak)
├── pyproject.toml                  # package metadata, deps, optional [ml] [dev]
├── src/autotokamak/                # the importable package
│   ├── core/                       # shared utilities (geometry, solver, io, diagnostics, logging, schema)
│   ├── agent/                      # URSA runners + prompts
│   │   ├── runners/                # plan_execute.py, plan_execute_feedback.py
│   │   └── prompts/                # YAML prompts the agent consumes
│   ├── data/                       # (Week 2) sweep generators, HDF5 loaders
│   ├── surrogate/                  # (Week 4) PINN, DeepONet, FNO, baselines
│   ├── models/                     # (Week 4+) trained-model loaders
│   └── eval/                       # (Week 3+) metrics, benchmarks, comparison plots
├── examples/                       # runnable demos, now built on autotokamak.core
│   ├── fixed_boundary/             # analytic + EQDSK demo (hardcoded physics)
│   └── config_driven_equilibrium/  # YAML-driven runner + sweep + ψ inverter
├── tests/                          # pytest suite (smoke + schema + geometry)
├── data/                           # gitignored: raw/, processed/ for training datasets
├── models/                         # gitignored: checkpoints/
├── experiments/                    # gitignored: per-experiment configs and logs
├── docs/                           # architecture diagrams and design notes
└── outputs/                        # gitignored: per-run artifacts from example scripts
```

## Two layers of code

### Layer 1 — Agent drivers (`src/autotokamak/agent/`)
These are the **agentic runners** that read a YAML prompt and let URSA do the work.

| File | What it does |
|---|---|
| `runners/plan_execute.py` | Plain plan → execute. PlanningAgent emits steps; ExecutionAgent runs each in turn, threading "previous-step summary" through the prompts. |
| `runners/plan_execute_feedback.py` | Same, plus a **feedback loop**: after execution, re-invoke the planner with the execution history so it can patch failures. Configurable via `feedback_rounds`, `validate_after`. |
| `runners/config.py` | Shared YAML loader and workspace-path resolver. |

Both runners:
- Load `.env` for `OPENAI_API_KEY`.
- `init_chat_model(model=...)` for both planner and executor (default `openai:o4-mini`).
- Create a workspace dir; symlink in `./ursa` and `./OpenFUSIONToolkit` so the agent can read them.

### Layer 2 — Generated example workspaces
These are **artifacts produced by Layer 1 agents** — concrete, hand-runnable TokaMaker examples. They have been committed to the repo so you can run them directly without invoking the agent.

| Dir | What it is | Key entry point |
|---|---|---|
| `examples/fixed_boundary/` | First demo: a fixed-boundary GS equilibrium with two cases (`analytic` vs `eqdsk`). Hardcoded physics; useful as a smoke test. | `python run_fixed_boundary_equilibrium.py --case analytic` |
| `examples/config_driven_equilibrium/` | More sophisticated: **fully config-driven** (no hardcoded `mesh_dx`, order, targets). Adds a discretization sweep runner and a ψ-inverter that tunes parameters to match a target flux map. | `python run_equilibrium_from_config.py discretization_config.yaml` |

## Prompts dir

`agent/prompts/*.yaml` — these are the inputs to Layer 1. Each contains:
- `problem:` — the natural-language task description given to the planner.
- `workspace:` — where the agent's outputs go (matches the Layer 2 dir names).
- `model:` — LLM string for `init_chat_model`.
- `symlinks:` — what to link into the workspace (always `./ursa` and `./OpenFUSIONToolkit`).
- Sometimes a `discretization_config_schema:` block that documents the expected YAML the agent should produce.

| Prompt file | Produced workspace | Purpose |
|---|---|---|
| `oft_example_generation.yaml` | `examples/fixed_boundary/` | "Build a fixed-boundary equilibrium example by reading the OFT notebook." |
| `oft_discretization_example.yaml` | `examples/config_driven_equilibrium/` | "Build a config-driven equilibrium example with a specified API surface." |
| `oft_example_configurable.yaml`, `_2.yaml` | (variant configs) | Tuning iterations of the above. |
| `example1.yaml` | `example1/` (not committed) | Earlier toy example. |

## How a typical run flows

```
agent/prompts/oft_discretization_example.yaml
        │
        ▼
python -m agent.runners.plan_execute --config agent/prompts/oft_discretization_example.yaml
        │
        ▼
PlanningAgent (LLM) reads problem → emits N steps
        │
        ▼
For each step:
   ExecutionAgent (LLM + tools) writes code, runs it, inspects output,
   passes summary to next step
        │
        ▼
Workspace (e.g. examples/config_driven_equilibrium/) populated with:
   - YAML config the agent generated
   - Python runner script
   - outputs/ dir with NPZ, JSON, PNG plots
   - README.md
```

After the agent finishes, the workspace is self-contained: you can re-run the example without involving any LLM at all.

## Setup

```bash
python3.11 -m venv venv && source venv/bin/activate

# Editable install of autotokamak with all dev tools
pip install -e ".[ml,dev]"

# OpenFUSIONToolkit binary + Python bindings (PyPI as of v26.6 — no /Applications install needed)
# Already included as a dependency in pyproject.toml; pip install above pulls it in.

# Optional: side-clones of OFT and URSA source for reference (not needed at runtime)
git clone https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit.git
git clone https://github.com/lanl/ursa.git

# Agent runners need OpenAI access:
echo 'OPENAI_API_KEY=sk-...' > .env
```

Python **must be 3.11 or 3.12** (some `ursa-ai==0.15.1` deps don't support 3.13+).

## What's gitignored

- `venv/`, `.env`
- `OpenFUSIONToolkit/`, `ursa/` (you side-clone these; they're not part of this repo)
- All build/cache dirs

## Physics one-liner

TokaMaker solves the **Grad–Shafranov equation**

```
Δ*ψ = −μ₀ R² p'(ψ) − F(ψ) F'(ψ)
```

on a 2D triangular mesh of a D-shaped plasma cross-section. Inputs: LCFS shape (R0, a, κ, δ), pressure/current profiles, total plasma current Ip. Output: flux function ψ(R,Z) plus derived quantities (q-profile, magnetic axis, etc.).

## Things to keep in mind when editing

- **Use `autotokamak.core`** for any geometry / solver / IO / logging logic. Don't duplicate it — extend the library.
- `examples/config_driven_equilibrium/run_equilibrium_from_config.py` is the **reference template** — config-driven, uses `core/`, extensible. Build new sweeps on this pattern.
- `examples/fixed_boundary/run_fixed_boundary_equilibrium.py` is a legacy first-pass demo. It still works but does not yet route through `core/`; treat it as a reference for the EQDSK-loading workflow.
- **OFT singleton**: only one `OpenFUSIONToolkit.OFT_env` can ever be created per Python kernel. `core.solver.make_solver` accepts an optional `env=` to reuse the existing one — required for any retry path or for batched solves in one process.
- Never write into side-cloned `./ursa/` or `./OpenFUSIONToolkit/` if you have them locally — they're read-only and gitignored.
- Agent prompts in `src/autotokamak/agent/prompts/*.yaml` contain hard `CONSTRAINTS:` blocks (no `git`, no `pip install`, no `input()`). Preserve these when editing.
- `outputs/`, `data/raw/`, `data/processed/`, `models/checkpoints/`, `experiments/` are all gitignored. Don't commit generated artifacts.
- Run `pytest tests/ -v` after structural changes; `pytest tests/ -v -m slow` to include the full OFT solve smoke test.
