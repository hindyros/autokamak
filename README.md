# autotokamak

ML surrogate models and agentic LLM workflows for the **Grad–Shafranov equation** — built on top of:
- **[OpenFUSIONToolkit (OFT)](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit)** — TokaMaker for ground-truth GS solves.
- **[URSA](https://github.com/lanl/ursa)** — LangChain/LangGraph agent framework for plan/execute workflows.

Built as a summer RA project at **MIT Energy Initiative**.

---

## Documentation

- [docs/architecture.md](docs/architecture.md) — high-level layering and data flow.
- [docs/agent-workflows.md](docs/agent-workflows.md) — how runners and prompts work.
- [docs/examples.md](docs/examples.md) — how to run and interpret example workspaces.
- [docs/configs.md](docs/configs.md) — agent task YAML vs simulation config YAML.
- [docs/development-notes.md](docs/development-notes.md) — migration notes and conventions.

---

## Setup (macOS / Linux)

```bash
python3.11 -m venv venv && source venv/bin/activate

# Editable install: pulls in OpenFUSIONToolkit, URSA, pydantic, h5py, etc.
pip install -e ".[ml,dev]"

# Agent runners need OpenAI access:
echo 'OPENAI_API_KEY=sk-...' > .env

# Optional: side-clone OFT and URSA source if you want to browse their examples
git clone https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit.git
git clone https://github.com/lanl/ursa.git
```

Python **must be 3.11 or 3.12**. OpenFUSIONToolkit (v26.6+) is on PyPI, so no
`/Applications/` install or `PYTHONPATH` exports are needed.

### Verify the install

```bash
python -c "from autotokamak.core import solver, geometry, schema; print('OK')"
pytest tests/ -v
```

---

## First example: Fixed-boundary equilibrium (OFT TokaMaker)

The **first example** in this repo is the **OpenFUSIONToolkit TokaMaker fixed-boundary equilibrium** workflow in `examples/fixed_boundary/`. It is a standalone Python script that:

- Builds and solves a **fixed-boundary Grad–Shafranov equilibrium** using OFT’s TokaMaker in fixed-boundary mode.
- Supports two cases:
  - **`--case analytic`**: the plasma boundary (LCFS) is generated analytically (e.g. an isoflux-shaped boundary).
  - **`--case eqdsk`**: the boundary is loaded from OFT’s bundled EQDSK example.
- For each run it: creates or reads the LCFS boundary, builds a GS domain mesh, configures TokaMaker with targets (e.g. total plasma current) and optional profiles, solves the equilibrium, and writes outputs (NPZ/JSON and optional PNG plots) under `examples/fixed_boundary/outputs/`.

**Quick run (from repo root, with venv active and OFT on `PATH`/`PYTHONPATH`):**

```bash
cd examples/fixed_boundary
python run_fixed_boundary_equilibrium.py --case analytic
```

---

## Agent workflows

Agent code lives under `src/autotokamak/agent/`:

- **`agent/runners/plan_execute.py`** — plan → execute loop using URSA's PlanningAgent + ExecutionAgent.
- **`agent/runners/plan_execute_feedback.py`** — same, with a re-planning feedback loop after failures.
- **`agent/prompts/*.yaml`** — task YAMLs (problem statement, workspace, model, symlinks).

Run from the repo root (with venv active):

```bash
python -m autotokamak.agent.runners.plan_execute \
  --config src/autotokamak/agent/prompts/oft_example_generation.yaml
```

---

## Links

- **URSA**: [github.com/lanl/ursa](https://github.com/lanl/ursa) — Universal Research and Scientific Agent.
- **OpenFUSIONToolkit**: [github.com/OpenFUSIONToolkit/OpenFUSIONToolkit](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit) — Open FUSION Toolkit (OFT) for plasma and fusion modeling.
