# Configuration Types

This repo uses two distinct YAML configuration categories.

## 1) Agent Task Configs (`agent/prompts/*.yaml`)

Used by URSA runners to define **what the agent should do**.

Common fields:
- `problem`
- `workspace`
- `model`
- `symlinks`

Example:
- `agent/prompts/oft_discretization_example.yaml`

Used with:

```bash
python -m agent.runners.plan_execute --config agent/prompts/oft_discretization_example.yaml
```

## 2) Simulation Physics/Discretization Configs (`examples/**.yaml`)

Used by example scripts to define **physics and numerical settings**.

Example:
- `examples/config_driven_equilibrium/discretization_config.yaml`

Used with:

```bash
python examples/config_driven_equilibrium/run_equilibrium_from_config.py examples/config_driven_equilibrium/discretization_config.yaml
```

## Rule of Thumb

- If it starts with `problem:` and mentions tasks/constraints, it is an **agent task** config.
- If it defines boundary, mesh, solver, targets, and outputs, it is a **simulation** config.
