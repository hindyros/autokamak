# Debugging triage

Symptom → cause → first move.

## Solver / OFT

### `RuntimeError: Only one instance of OFT_env can be created per Python kernel`

**Cause:** something called `OpenFUSIONToolkit.OFT_env(...)` directly, or a second `import` path bypassed the `_OFT_ENV_CACHE`.

**Fix:** replace direct construction with `autotokamak.core.solver.get_oft_env()`. If in a retry path, pass `env=` to `make_solver`. If you truly need a fresh environment, spawn a subprocess.

### `used_fallback=True` (from `get_last_solve_info()`)

**Cause:** `set_isoflux(...)` or `solve()` raised at the isoflux-constrained solve; the retry path solved unconstrained.

**First check:**
- Sample shape outside the feasible box? Run `scripts/probe_feasible.py` and pick bounds where isoflux-used fraction is high.
- `init_psi.method: isoflux` failing? Try `tokamaker_default`.
- κ > 1.6 or δ > 0.4? Feasibility drops sharply — expect it.

**Impact:** the saved ψ no longer matches the geometry inputs faithfully. For a **single solve**, note it in the summary. For a **dataset sweep**, silently keeping fallback samples poisons the surrogate — either mask them, or regenerate cleanly with tighter bounds.

### `q_95` is `NaN` or extreme

**Cause:** the safety-factor extractor hit a boundary inside `gs.get_q()`, or the equilibrium didn't fully converge.

**First check:**
- `mesh_dx` too coarse? (>0.02 often produces bad q profiles). Try 0.015 or 0.010.
- `solver.maxits` too low? Bump to 100 for shape edge cases.
- Extreme κ or δ? See feasible-box comment above.

### Mesh build fails at `add_polygon`

**Cause:** the LCFS polygon self-intersects. Usually too-few `npts` for a curvy shape.

**Fix:** `boundary.npts >= 40` (default 80). If `delta > 0.3`, prefer 80+.

## Data / IO

### HDF5 dataset opens but `psi` is all zero, or reads truncated

**Cause:** dataset generator crashed mid-write and the atomic rename never fired.

**First check:** ls the `outputs/` dir for temp files (numpy leaves `tmp*.npz`). Delete the partial `.h5` and re-run. If you're using `autotokamak.core.io.atomic_savez` this shouldn't happen — inspect the traceback in `experiments/<run_id>/trace.json` for the real crash.

### `assert_nonempty_file` raises with size < 16 bytes

**Cause:** downstream wrote an empty file. Almost always a masked exception in the writer. Grep the runner log for "WARNING" or read `experiments/<run_id>/trace.json`.

## Agent runners

### `plan_execute_feedback` loops without progress

**First check:**
- `scorer` set but no `expected_artifacts`? The scorer's hard gates never pass without artifact deliverables. Add both.
- Prompt CONSTRAINTS violation (agent tried `pip install` or `git`)? Look for `WORKSPACE HYGIENE WARNING` blocks in stdout — the runner emits these into round-N+1's planner prompt.
- Model returning empty plans? Check `--model`; some smaller models drop the plan structure.

### `meta_loop` action dispatch fails

**First check:**
- `initial_dataset_h5` missing? The path is repo-root relative.
- `base_sweep_config` missing? Same.
- Optimized DSPy checkpoint at `agent/dspy/optimized/meta_picker.json` corrupt? Add `--use-baseline` to force the in-code baseline picker.

### Trace file at `experiments/<run_id>/trace.json` is empty/missing

**First check:**
- Runner given `--no-trace`? Remove it.
- `experiments/` dir not writable? Check permissions.
- Runner crashed before `RunTrace.open()` — check stderr; the trace open is inside a try/except so open failures don't abort the run.

## Environment

### `import OpenFUSIONToolkit` fails

**First check:** `pip install "OpenFUSIONToolkit>=26.6"` (or the pinned version in `pyproject.toml`). Requires Python 3.11 or 3.12. On 3.13+, `ursa-ai==0.15.1` deps also break.

Run `scripts/check_env.py` to get a one-line PASS/FAIL summary.

### `ImportError: cannot import name 'gs_Domain'` (or similar OFT API drift)

**Cause:** installed OFT version doesn't match what the runner expects.

**First check:** `pip show OpenFUSIONToolkit`. autotokamak's core is best-effort tolerant to API drift (see `_try` helpers in `diagnostics.py`) but `geometry.py` and `solver.py` require specific entry points. If OFT bumped a major version, a small shim in `core/geometry.py` or `core/solver.py` is the right fix.

## When the fix belongs in the platform, not the workspace

If the bug is in a *generated* runner (`examples/surrogate_automl/run.py`, `examples/dataset_generation/run_dataset_sweep.py` when it was produced by the agent, or anything under `experiments/**`):

- **DO** edit the corresponding **prompt YAML** (`src/autotokamak/agent/prompts/*.yaml`) or the underlying runner (`src/autotokamak/agent/runners/*.py`) or the scorer (`src/autotokamak/agent/dspy/metric_*.py`).
- **DO NOT** patch the generated file in place. It will be overwritten on the next agent run, and the fix won't propagate to future runs.

This rule is load-bearing. See `feedback_dont_edit_agent_output` in memory.
