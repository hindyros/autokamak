# Examples Guide

This document covers runnable simulation content under `examples/`.

## `examples/fixed_boundary`

Purpose:
- Demonstrates a fixed-boundary Grad-Shafranov solve with OpenFUSIONToolkit/TokaMaker.
- Includes analytic and EQDSK boundary modes.

Main script:
- `examples/fixed_boundary/run_fixed_boundary_equilibrium.py`

Quick run:

```bash
python examples/fixed_boundary/run_fixed_boundary_equilibrium.py --case analytic
```

## `examples/config_driven_equilibrium`

Purpose:
- Config-driven equilibrium workflow with discretization controls.
- Includes sweep and inversion helpers.

Main script:
- `examples/config_driven_equilibrium/run_equilibrium_from_config.py`

Quick run:

```bash
python examples/config_driven_equilibrium/run_equilibrium_from_config.py examples/config_driven_equilibrium/discretization_config.yaml
```

## Outputs

Both examples write timestamped or hashed run artifacts under each example's local `outputs/` directory.
