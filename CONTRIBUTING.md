# Contributing

The most useful thing you can add is **a new agent substrate**. That is what
the benchmark exists to compare, and the interface is small. This document
spells out what it actually obliges you to do, because the abstract method
alone does not tell you.

## Adding a harness

Two steps in principle:

1. A module in `src/autotokamak/harnesses/` with a class deriving from
   `Harness`.
2. One line in `src/autotokamak/harnesses/registry.py`.

`tests/test_harness_contract.py` will pick it up automatically and check the
mechanical parts. Run it first:

```bash
pytest tests/test_harness_contract.py -q
```

Read `src/autotokamak/harnesses/echo.py` before you start. It is 77 lines, it
uses no language model, and it satisfies every obligation below — which makes
it the reference implementation.

### The contract, in full

```python
class MyHarness(Harness):
    name = "mine"          # a ClassVar. The base class only *annotates* it,
                           # so forgetting it fails deep inside condition_for()
                           # rather than at class definition.

    def run(self, task, workspace, *, run_dir, model=None,
            timeout_seconds=None) -> RunResult:
        ...
```

Your `run` must:

- **Materialise `task.symlinks`** into the workspace before the agent starts.
  The L3 tasks link a read-only copy of the solver source tree; without it the
  agent cannot do the task.
- **Write a `RunTrace`** (`autotokamak.bench.trace`) under `run_dir`, mapping
  your substrate's native events onto its round/step structure. This is what
  the methodology extraction and the HTML report read.
- **Confine all writes** to `workspace` and `run_dir`. Nothing else.
- **Leave the raw event stream** at `run_dir/<name>_events.jsonl`, unmodified.
  It is the evidence of last resort when a run does something surprising.
- **Honour `timeout_seconds`.** How depends on your substrate:
  - *In-process*: wrap the agent loop in `base.time_limit()`. Read that
    function's docstring — `HarnessTimeout` inherits `BaseException` for the
    same reason `KeyboardInterrupt` does, because a framework's broad
    `except Exception` will otherwise swallow the deadline and, since
    `signal.alarm` is one-shot, disable it permanently. One substrate ran for
    ten hours against a 90-minute cap before we understood this.
  - *Subprocess*: start it with `start_new_session=True` and signal the
    **process group** on timeout. A plain `subprocess.run(timeout=...)` kills
    the direct child and then blocks forever waiting on pipes that the
    solver's grandchildren still hold open.
- **Report cost** in `RunResult.cost_usd` if your substrate exposes it. If it
  does not, leave it `None` and `tools/cost_report.py` will derive a figure
  from token counts and the committed price table — but label it derived, as
  we do for the substrates that bill against their own quota.

Optional overrides, all with sensible defaults in `base.py`: `condition_for`,
`resolve_model`, `prepare_workspace`, `workspace_note` (the shared
working-directory instruction appended to every prompt — keep it identical
across substrates, it is part of what makes them comparable), and
`dry_run_info`.

### Model naming

Substrates disagree about how a model is named: some want `openai:gpt-5.2`,
some `openai/gpt-5.2`, some a bare `gpt-5.2`. Do not paper over it in the
adapter. Declare what yours takes, and let the task file pin it per adapter —
that way the model string lands in the run record three separate times and the
control is auditable afterwards.

## Things that are frozen

Changing any of these silently invalidates every archived result, so they are
not to be edited without versioning the change:

- `src/autotokamak/bench/contract.py` — the gate set and their semantics.
- `benchmarks/assets/eval_grid.json`, `test_params.json` — the evaluation grid
  and the held-out parameters.
- `benchmarks/assets/test_set.h5` — never regenerate in place. It carries
  provenance attributes, and every archived score is defined relative to it.
- The `problem:` text of any task YAML that has been run. Bump
  `prompt_version` instead; the aggregation refuses to pool across versions.

New measurements belong in `bench/diagnostics.py` as **recorded values, not
gates**. Adding a gate changes `contract.passed` and makes old runs
incomparable.

## Before you open a pull request

```bash
ruff check src tools tests
pytest tests/ -q
```

Both must be clean; CI runs exactly these on 3.11 and 3.12. If you touched
anything on the measurement path, also re-aggregate a campaign and confirm the
CSVs are byte-identical:

```bash
python tools/aggregate_matrix.py --tag <a tag you have>
```

## A note on the provenance comments

Nearly every file starts with a line like:

```python
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
```

This is not decoration and it is not an apology. This repository contains two
kinds of code, and a benchmark that measures agent-written code has to be able
to tell them apart:

- **Platform code** — the solver wrappers, the contract, the harnesses, the
  analysis. Human-designed, LLM-assisted in the writing. It is the measuring
  instrument, not the thing being measured.
- **Agent-generated code** — marked `URSA-generated` and confined to
  `examples/`. This is the object of study, and its provenance is a
  scientific fact about the artifact.

If you add a file, mark it honestly.
