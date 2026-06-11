# CLAUDE.md — Repo guide for agentic_simulations

## What this repo is

A demo / scaffolding repo that connects two LANL-adjacent tools:

- **URSA** ([lanl/ursa](https://github.com/lanl/ursa)) — *Universal Research and Scientific Agent*. A LangChain/LangGraph-based agent framework with `PlanningAgent` + `ExecutionAgent`. It plans multi-step research tasks, then drives an executor that can write code, run shell commands, and inspect outputs.
- **OpenFUSIONToolkit (OFT)** ([OpenFUSIONToolkit/OpenFUSIONToolkit](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit)) — plasma/fusion modeling, including **TokaMaker** which solves the Grad–Shafranov equation for tokamak MHD equilibria.

The point: let an LLM agent autonomously plan and run a TokaMaker simulation end-to-end (build LCFS → mesh → solve GS → postprocess), with the human only providing a high-level problem statement and a YAML config.

## Two layers of code

### Layer 1 — Agent drivers (`agent/`)
These are the **agentic runners** you actually invoke. They read a YAML prompt and let URSA do the work.

| File | What it does |
|---|---|
| `agent/runners/plan_execute.py` | Plain plan → execute. PlanningAgent emits a list of steps; ExecutionAgent runs each in turn, threading "previous-step summary" through the prompts. |
| `agent/runners/plan_execute_feedback.py` | Same, but with a **feedback loop**: after each execution round, re-invoke the planner with the execution history so it can patch failures or confirm completion. Configurable via `feedback_rounds`, `validate_after`. |
| `agent/runners/config.py` | Shared YAML config loading and workspace path resolution (relative to repo root). |

Both runners:
- Load `.env` for `OPENAI_API_KEY`.
- `init_chat_model(model=...)` for both planner and executor (default `openai:o4-mini`; YAML or `--model` overrides).
- Create a `workspace/` dir; optionally symlink in `./ursa` and `./OpenFUSIONToolkit` so the agent can read them.
- Invoke: `python -m agent.runners.plan_execute --config agent/prompts/oft_example_generation.yaml`.

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

## Setup (from README, condensed)

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo 'OPENAI_API_KEY=...' > .env

# Side-clones (not in this repo; gitignored)
git clone https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit.git
git clone https://github.com/lanl/ursa.git

# OFT binary on PATH + PYTHONPATH (zshrc)
export OFT_ROOTPATH="/Applications/OpenFUSIONToolkit"
export PATH="$OFT_ROOTPATH/bin:$PATH"
export PYTHONPATH="$OFT_ROOTPATH/python:$PYTHONPATH"
```

Python **must be < 3.13** (some `ursa-ai==0.15.1` deps don't support 3.13+).

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

- **Never write into `./ursa/` or `./OpenFUSIONToolkit/`** — they're treated as read-only by agent prompts and are gitignored.
- The agent prompts contain hard `CONSTRAINTS:` blocks (no `git`, no `pip install`, no `input()`). Preserve these when editing prompts.
- `examples/config_driven_equilibrium/` has the more reusable architecture; `examples/fixed_boundary/` is a first-pass demo and the two are not meant to share code.
- Outputs are timestamped under each example's `outputs/` dir — safe to delete.
