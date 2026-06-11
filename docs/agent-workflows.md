# Agent Workflows

This document covers the `agent/` subtree only.

## Directory Map

- `agent/prompts/`: YAML prompts with `problem`, `workspace`, `model`, and `symlinks`.
- `agent/runners/config.py`: shared config loading and workspace path resolution.
- `agent/runners/plan_execute.py`: one-shot plan then execute.
- `agent/runners/plan_execute_feedback.py`: iterative re-plan and execute loop.

## Typical Run

From repository root:

```bash
python -m agent.runners.plan_execute --config agent/prompts/oft_example_generation.yaml
```

Feedback variant:

```bash
python -m agent.runners.plan_execute_feedback --config agent/prompts/oft_discretization_example.yaml
```

## Prompt Conventions

- `workspace` should point to a path under `examples/` for generated simulation workspaces.
- `symlinks` usually includes read-only links to `./ursa` and `./OpenFUSIONToolkit`.
- Keep `CONSTRAINTS` blocks explicit and unchanged unless intentional.
