# `tools/` — post-run analysis and diagnostics scripts

Standalone scripts that read the artifacts a pipeline run leaves behind.
None of them are needed to *run* the pipelines — the front door for that is:

```bash
python -m autotokamak.pipelines <phase1|phase2|meta> [--level L0|L1]
```

(Agent-codegen benchmark conditions run via
`python -m autotokamak.bench run --task benchmarks/tasks/<task>.yaml --harness <name>`;
see `benchmarks/README.md`.)

| Script | What it does |
|---|---|
| `eval_surrogate.py` | 7 diagnostic PNGs + JSON metrics for a trained surrogate (default workspace: `examples/surrogate_meta`). |
| `render_meta_plots.py` | Meta-loop convergence + per-cell RMSE plots from `meta_trace.json` / `report.json`. |
| `render_physics.py` | ψ(R,Z) contour samples + parameter histograms from a `dataset.h5`. |
| `trace_to_html.py` | Renders `experiments/*/trace.json` into a browsable static HTML report (stdlib-only). |
| `probe_feasible_box.py` | LHS-samples candidate shaping-parameter boxes and reports the clean-isoflux success rate per box. |
| `collect_traces.sh` | Runs the meta-loop N times to collect traces for offline GEPA prompt optimization (`agent/dspy/optimize_meta.py`). |
| `matrix_report.py` | Cross-condition matrix report: scores every cell (bench `predict.py` or pipeline `winner.pkl`) on the frozen benchmark test set; renders `experiments/<tag>/index.html`. |
| `eval_code_metrics.py` | Static (zero-LLM) metrics over every agent workspace in a tag: SLOC, structure, imports (library leverage / L3-violation cross-check / LLM-in-loop detection), seeds, ruff counts → `<run_dir>/eval/code_metrics.json` + `experiments/<tag>/code_metrics.csv`. |
| `judge_code.py` | Blind, outcome-blind LLM-as-judge over agent workspaces: 7-dimension rubric (1–5, evidence-required) per run (`score`), plus a cross-cell synthesis (`compare` → `judge_report.md`). Anonymized bundles, brand tokens redacted; `--samples N` for median-of-N. |
| `campaign_guard.py` | Guards around a paid campaign: `preflight` (keys, CLIs, model pins, frozen assets, disk, git state, and a refusal to include a non-OpenAI harness), `completed` (backs `run_campaign.sh --resume` so a crash does not re-pay for finished cells), `spend` (dollars so far, including cost recovered from killed runs' `ursa_metrics/`), `reconcile` (stub `result.json` for watchdog-killed cells, which otherwise vanish from every report), `forecast` (worst-case spend and wall clock per substrate, honouring rep caps / harness budgets / harness timeouts), `ratelimits` (this key's real TPM/RPM from response headers, sized against measured per-session consumption). |
| `aggregate_matrix.py` | **Replicate-aware** cell summary for a campaign: groups `<tag>/*/*/result.json` by condition → n, contract pass-rate (Wilson CI), rel-L2 median (bootstrap CI), physical-validity and passed-but-invalid rates, honesty gap, cost; pools by access level and runs the L2-vs-L3 contrast paired within harness. Also emits the **methodology matrix** (modal chain of methods per cell, chain agreement across replicates, acquisition criteria, per-round decision logic; `--show-rounds` prints every round) the **code comparison** (what the generated code's chooser actually computes, whether it calls the surrogate, and stated-vs-implemented agreement), and the **solution cross-comparison** (`solution_shape.csv`: one row per demand of the task — OFT_env ownership, solve isolation, mesh route, grid mapping, masking rule, storage + validation, pilot gate, leakage, self-test, code shape — one column per cell, disagreements first) and writes `methodology.csv` + `methodology_rounds.csv`. Refuses to pool prompt versions. Produces the paper's main table. |
| `run_campaign.sh` | Campaign driver with **replicates**: runs each (level, harness) cell `--reps` times in level-ordered waves at `--parallel` N, one log + exit code per run. `--model` for a model-controlled arm, `--dry-run` to cost nothing. |
| `cost_report.py` | Per-run cost/efficiency table for a tag: measured $ (claude_sdk/pi/dspy/ursa), token-derived $ (cursor + price table), wall-clock/turn proxies, solver-call counts. Harvests old runs' raw event streams retroactively. Loads `benchmarks/assets/prices.json` automatically and labels any model it cannot price as `unpriced:<model>` rather than leaving a blank. |

Each script's module docstring documents its exact CLI; run with `--help` for flags.
