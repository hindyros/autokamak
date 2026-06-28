# `autotokamak.agent.dspy`

DSPy integration scaffolding. Branch `dspy-integration`, status proposal.

See [`../../../../docs/dspy_integration_plan.md`](../../../../docs/dspy_integration_plan.md) for the full plan, options, and decision points.

## What's here today

| File | Purpose |
|---|---|
| `metric.py` | Composite scoring function for one agent run. Pure-Python, no DSPy dependency. Run today against any `examples/dataset_generation/`-shaped workspace. |
| `__init__.py` | Re-exports `score_run`, `ScoreReport`. |

## Quick start (works without DSPy installed)

```python
from autotokamak.agent.dspy import score_run

report = score_run("examples/dataset_generation/", requested_n_samples=16)
print(report.summary())
# total in [0, 1]; 0 if any hard gate failed
```

You can run this against the current dataset to confirm the metric flags the
known issues (it should give a moderate score with `inside_lcfs_quality < 0.2`,
catching the `griddata(nearest)` silent-fill bug).

## What is NOT here yet (intentionally)

- `linter.py` — DSPy module for prompt-quality prediction (Option C).
- `repair.py` — DSPy module for the feedback-loop replanner (Option B).
- `planner.py` — full-pipeline DSPy planner (Option A).
- Run-instrumentation patch to the runners.

These are gated on the run-instrumentation work and on accumulating ≥10 labeled
traces. See plan §3 and §5.

## Why this scaffolding is empty on purpose

DSPy optimizers (BootstrapFewShot, MIPROv2) need a training corpus. We have
~3 historical runs. Writing the DSPy modules before the corpus exists wastes
the optimizer's main advantage and locks in an API choice we can't validate.

The metric, on the other hand, can be exercised today — and a well-defined
metric is the load-bearing piece for any of Options A/B/C.
