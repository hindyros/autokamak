# URSA + OpenFUSIONToolkit Examples

This repository is an example of using the **[Universal Research and Scientific Agent (URSA)](https://github.com/lanl/ursa)** to run simulations with the **[Open FUSION Toolkit (OFT)](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit)**. URSA provides agentic workflows for planning, code execution, and research; OFT provides modeling tools for plasma and fusion in 2D/3D (including TokaMaker for MHD equilibria).

---

## Documentation

- [docs/architecture.md](docs/architecture.md) — high-level layering and data flow.
- [docs/agent-workflows.md](docs/agent-workflows.md) — how runners and prompts work.
- [docs/examples.md](docs/examples.md) — how to run and interpret example workspaces.
- [docs/configs.md](docs/configs.md) — agent task YAML vs simulation config YAML.
- [docs/development-notes.md](docs/development-notes.md) — migration notes and conventions.

---

## Setup (macOS)

1. **Create a virtual environment and install Python dependencies**

   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Create a `.env` file** in the repo root and add your OpenAI API key (required for URSA workflows that use an LLM):

   ```bash
   echo 'OPENAI_API_KEY=your-key-here' > .env
   ```

   Replace `your-key-here` with your key from [OpenAI API keys](https://platform.openai.com/account/api-keys). The `.env` file is gitignored.

3. **Clone OpenFUSIONToolkit and URSA** (if not already present)

   ```bash
   git clone https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit.git
   git clone https://github.com/lanl/ursa.git
   ```

4. **Install the OpenFUSIONToolkit binary** and set up your environment so the OFT tools and Python bindings are on your `PATH` and `PYTHONPATH`. For a typical macOS install under `/Applications/OpenFUSIONToolkit`:

   ```bash
   echo 'export OFT_ROOTPATH="/Applications/OpenFUSIONToolkit"' >> ~/.zshrc
   echo 'export PATH="$OFT_ROOTPATH/bin:$PATH"' >> ~/.zshrc
   echo 'export PYTHONPATH="$OFT_ROOTPATH/python:$PYTHONPATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

   Adjust `OFT_ROOTPATH` if you installed OFT elsewhere. After this, the `OpenFUSIONToolkit` Python module and OFT binaries (e.g. TokaMaker) should be available in your shell.

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

All agent-related code lives under `agent/`:

- **`agent/runners/plan_execute.py`** — plan → execute loop using URSA’s Planning and Execution agents.
- **`agent/runners/plan_execute_feedback.py`** — same, with a re-planning feedback loop after failures.
- **`agent/prompts/`** — YAML task configs (problem statement, workspace, model, symlinks).

Run from the repo root (with venv active):

```bash
python -m agent.runners.plan_execute --config agent/prompts/oft_example_generation.yaml
```

See `agent/prompts/oft_discretization_example.yaml` for the more advanced config-driven equilibrium task.

---

## Links

- **URSA**: [github.com/lanl/ursa](https://github.com/lanl/ursa) — Universal Research and Scientific Agent.
- **OpenFUSIONToolkit**: [github.com/OpenFUSIONToolkit/OpenFUSIONToolkit](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit) — Open FUSION Toolkit (OFT) for plasma and fusion modeling.
