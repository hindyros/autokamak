# Development Notes

## Current Layout Convention

- Agent orchestration code lives in `agent/`.
- Runner entrypoints live in `agent/runners/`.
- Agent prompts live in `agent/prompts/`.
- Runnable simulation workspaces live in `examples/`.

## Migration Notes

Repository paths were refactored from:
- `oft_generation_example/` -> `examples/fixed_boundary/`
- `oft_discretization_example/` -> `examples/config_driven_equilibrium/`
- `agent/plan_execute.py` -> `agent/runners/plan_execute.py`
- `agent/plan_execute_feedback.py` -> `agent/runners/plan_execute_feedback.py`
- `agent/config.py` -> `agent/runners/config.py`

## Keep In Mind

- Do not modify side clones `OpenFUSIONToolkit/` and `ursa/`.
- Keep prompt `workspace:` values aligned with paths under `examples/`.
- Keep agent and simulation concerns separate in docs and new files.
