# Benchmarks — the agent-capability experiment matrix

Can a coding agent execute the 4-step surrogate pipeline this repo implements
— (1) generate Grad-Shafranov data, (2) feature-engineer, (3) train
surrogates, (4) decide what to do next — and how much does pre-written,
proven library code help? Every experiment in this repo is one cell of a
2-axis matrix, named `<level>-<harness>`:

## Axis B — access level

| Level | Meaning | Where it runs |
|---|---|---|
| **L0** scripted | Pre-written pipeline, seeded heuristic decisions, zero LLM. The reproducible golden baseline. | `python -m autotokamak.pipelines <phase> --level L0` |
| **L1** structured | Pre-written pipeline; an LLM makes only typed decisions (search rounds, meta actions) via the DSPy pickers. | `python -m autotokamak.pipelines <phase> --level L1` |
| **L2** library-assisted | The agent writes the glue code and MAY import `autotokamak`. | `python -m autotokamak.bench run --task benchmarks/tasks/L2_library.yaml --harness <name>` |
| **L3** from-scratch | The agent writes everything; importing `autotokamak` is forbidden and AUDITED (hard contract gate). | `python -m autotokamak.bench run --task benchmarks/tasks/L3_from_scratch.yaml --harness <name>` |

## Axis A — harness (agent substrate)

`ursa` (URSA PlanningAgent+ExecutionAgent) · `dspy` (DSPy plan→ReAct→review) ·
`claude_sdk` (Claude Agent SDK) · `pi` (Pi Code CLI) · `cursor` (Cursor CLI) ·
`echo` (no-LLM mock for CI). L0 has no agent (`L0-none`); L1 is DSPy-typed by
construction (`L1-dspy`).

## The shared contract

Every condition emits the same machine-checkable deliverables, so all cells
are directly comparable (`src/autotokamak/bench/contract.py`):

- `report.json` with `n_solves_*` and `metrics.{test,baseline}_rel_l2.{mean,median,p90}`
- `predict.py --input params.json --output pred.npz` with `psi (N, 96, 64)`
  on the frozen grid (`assets/eval_grid.json`: R 64 pts in [0.15, 0.80] m,
  Z 96 pts in [−0.40, 0.40] m; physical psi in Wb, NaN outside the plasma)
- head-to-head scoring against the frozen test set: `assets/test_params.json`
  (60 params, seed 20260809, committed) solved once by
  `python -m autotokamak.bench freeze-testset` into `assets/test_set.h5`
  (gitignored, reproducible)

## Running a condition

```bash
# 1. cheap smoke first — auth, jailing, trace capture, exit handling:
python -m autotokamak.bench run --task benchmarks/tasks/smoke.yaml --harness claude_sdk
# 2. see exactly what would run without spending anything:
python -m autotokamak.bench run --task benchmarks/tasks/L3_from_scratch.yaml --harness cursor --dry-run
# 3. the real campaign:
python -m autotokamak.bench run --task benchmarks/tasks/L3_from_scratch.yaml --harness ursa --tag aug09
# 4. compare every run under a tag:
python -m autotokamak.bench compare --tag aug09
```

Each run writes `experiments/<tag>/<condition>/<run_id>/{workspace/,
trace.json, result.json}`; `result.json` bundles the harness outcome, the
contract gates, and (when the frozen test set exists) the head-to-head score.

## Diagnostics beside the contract

`contract.passed` is a frozen comparability asset — adding a gate would
silently make every archived run incomparable. So the checks that catch what
the gates miss live in `src/autotokamak/bench/diagnostics.py` and are
recorded as a `diagnostics` block in `result.json`, never as gates:

- `test_rel_l2_full_grid` and `exterior_inflation` — the contract metric
  masks to finite ground truth, so field predicted *outside* the plasma is
  scored as if absent. The full-grid variant charges for it; the ratio of
  the two is the violation's size in physical terms. This is the
  discriminator: two shakedown runs both scored NaN-mask agreement 0.195,
  but one had inflation 1.22 (a NaN-convention slip on an otherwise good
  model) and the other 16.97 (a fabricated exterior plasma whose honest
  error is 1.94, four times worse than baseline). Counting disagreeing
  pixels cannot tell those apart; magnitude can, which is why
  `physically_valid` is judged on magnitude and `nan_mask_agreement` is
  recorded but not decisive.
- `pred_over_truth_spread` — catches a constant predictor that ignores its
  inputs.
- `physically_valid` and `passed_gates_but_invalid` — the headline
  measurement: green on every machine-checkable gate, wrong in the field.
- `honesty_gap` — agent's self-reported `metrics.test_rel_l2.mean` minus the
  independently scored value, signed (negative = claimed better than it is).
  `report_keys` only ever checked that metric keys were *present*, which is
  exactly what makes this measurable. The same L3-ursa run reported 0.0784
  against an actual 0.9596.
- solver-budget use against v3's ≤150 + 3×100 campaign bound, and
  acquisition-log presence.

Thresholds are named constants in that module. Backfill the block onto older
runs with `python -m autotokamak.bench diagnose --tag <tag> [--rescore]`
(`--rescore` re-invokes each `predict.py`, costing CPU but no API spend).

## Methodology: what was chosen, and on what reasoning

Every measurement above scores the OUTCOME. None of them separates two runs
that reach the same error by different reasoning — ensemble disagreement with
a stop triggered by measured validation error, versus uniform random sampling
relabelled "adaptive" and stopped because the rounds ran out. That difference
is this repo's actual research question, so it is extracted as data, not left
as prose in a README.

`src/autotokamak/bench/methodology.py` is deterministic, LLM-free and
execution-free: it reads the artifacts the v3 task already mandates (the
acquisition log, `sampling_strategy`, `report.json`) plus the agent's own
code, and writes a `methodology` block into `result.json`. Two things come
out of it:

1. **The chain of methods** — the final pipeline, canonicalised into one
   vocabulary: `initial design → representation + model [ensembling] →
   acquisition × rounds → stopping rule`, e.g.
   `lhs -> pca+mlp_torch[mc_dropout] -> acq:uncertainty_ensemble x3 -> stop:val_threshold_70pct`.
2. **The per-iteration decision logic** — for each adaptive round: the
   criterion the agent stated, how many points it took, the validation error
   against baseline it had *in hand* when it chose, and what it decided next
   (`continue` / `stop_threshold_met` / `stop_without_threshold` /
   `stop_unexplained`).

The acquisition vocabulary is anchored on the L0/L1 typed action space
(`agent.orchestrator.schema.AcquisitionStrategy`), and
`extract_meta_methodology` maps a pipeline workspace's `meta_trace.json` onto
the same record — so a scripted L0 policy, an L1 typed picker and a
from-scratch L3 agent are described in one vocabulary and land in one table.

Derived measurements that the score cannot give:

- `chain_agreement` — share of a cell's replicates on the modal chain. Method
  reproducibility is separate from score reproducibility: a cell can be
  stable in error while its agent re-invents the pipeline every run.
- `criterion_switched` — the acquisition criterion CHANGED between rounds
  (logic that reacts to what it measured) rather than one fixed rule executed
  n times.
- `evidence_grounded` — fraction of rounds whose validation-vs-baseline error
  was actually recorded. An ungrounded round's reasoning is unfalsifiable.
- `adaptive_in_name_only` — every stated criterion names nothing but
  randomness.
- `prose_only_terms` — a method claimed in README/`report.json` that the code
  never evidences (matrix-v3 shakedown: URSA's README claims ensemble
  uncertainty; its log shows `farthest_point` on all three rounds).

### The code comparison

Everything above reads the agent's own account of itself — a log's "reason",
a README's prose. `analyse_code_logic` reads the decision CODE instead. It
parses every agent-authored `.py`, locates the functions that choose the next
batch (by name: `acquire`/`select`/`propose`/`score`/`uncertainty`/…, falling
back to file scope for straight-line scripts), and classifies the arithmetic
each one actually performs — into the same vocabulary — with `file:line`
evidence for every claim:

- `code_acq` — the criterion the chooser computes: a standard deviation over
  stacked model predictions, a `cdist` to the training set, `rng.choice`.
- `model_informed` — whether that chooser calls the surrogate at all. This is
  load-bearing: model-derived criteria (uncertainty, residual) are DROPPED
  when it does not, because a `.std()` used to standardise parameters is
  indistinguishable from one ranking predictive spread until you ask whether
  the model was consulted. URSA's `propose_adaptive_batch` standardises with
  `.std()` and never predicts — it is farthest-point, and is recorded as
  such, whatever its docstring calls it.
- `selection_rule` — `top_k` (argsort/topk) vs `random_draw` vs `threshold`.
- `stated_vs_code` — the comparison column: `agree` / `partial` / `mismatch` /
  `unverifiable_from_code` / `undocumented`, with `only_stated` (claimed but
  never computed) and `only_implemented` (computed but never claimed).
  Compared at the level of criterion FAMILY, not exact formula. A random
  candidate POOL is discounted when the chooser also ranks: nearly every
  implementation draws one, and it is not the selection criterion.

Per cell the matrix then carries `code_logic_modal`, `code_logic_agreement`
(replicates implementing the same logic), `n_distinct_code_logics`,
`model_informed_k/n`, `stated_vs_code` verdict counts and
`claimed_not_implemented`.

### How the prompt was solved — the cross comparison

The acquisition criterion is one paragraph of a task that also asks the agent
to drive a finite-element solver under a one-`OFT_env`-per-process
constraint, run and validate a data campaign, decide what "no plasma here"
means when writing NaN, map a triangular mesh onto a frozen rectangle, and
ship a CLI a stranger can run. Two agents can share an acquisition criterion
and have solved almost none of those the same way.

`src/autotokamak/bench/solution_shape.py` answers each of those demands from
the code that plays that ROLE — the file that constructs `OFT_env`, the
predictor that writes the NaNs, the function that calls `.fit()` — never a
workspace-wide grep, which in this corpus is true of everything and
therefore says nothing. Every answer carries `file:line`.

| dimension | the question it answers |
|---|---|
| `oft_env_strategy` | OFT allows one `OFT_env` per process, ever. Who owns it? |
| `solve_isolation` | What insulates one solve from the next? |
| `mesh_route` | The OFT API, or a hand-built triangulation (which the task forbids)? |
| `grid_mapping` | How does the mesh reach the frozen 64×96 grid? |
| `mask_rule` | How is "no plasma here" decided when writing NaN? |
| `storage` | What is a solved sample on disk, and is it indexed? |
| `storage_validation` | Is a solve counted only after its file re-loads finite? |
| `pilot_gate` | Was the mandated pilot run, and its 50% threshold enforced? |
| `leakage_guard` | Does the test set stay out of the functions that fit the model? |
| `self_test` | Was the documented `predict.py` CLI re-run in a fresh process? |
| `code_shape` / `entry_point` | One script or a module tree; how would a stranger run it? |

Two scoping rules do most of the work, and both were added after they went
wrong on real runs:

- **The production path is the default scope.** The task MANDATES a meshing
  milestone, a pilot and a deliverable self-test, so every workspace contains
  demo and verification scripts. Reading the campaign's execution model off a
  `final_repro_check.py` is how a cross-comparison becomes fiction, so
  `smoke|milestone|preflight|parity|repro|tests/` are excluded — except for
  `self_test`, which is *supposed* to live in such a script.
- **Function scope, not file scope.** "The file that trains also mentions
  test" is true of any single-file pipeline; "the function that calls
  `.fit()` touches a test path" is a specific thing to go and read.

`cross_compare` transposes it — one row per dimension, one column per cell,
rows the agents disagreed on first — which is what a comparison has to look
like to be read as one. It prints in `tools/aggregate_matrix.py`, lands in
`solution_shape.csv`, and heads the HTML report with `file:line` on hover.

The v3 shakedown, four agents on an identical prompt: all four meshed through
`gs_Domain`, evaluated ψ with the solver's own `get_field_eval`, enforced the
pilot gate and re-loaded stored artifacts before counting them — and they
disagreed on execution model (`in_process_serial` vs `subprocess_per_batch`
vs `process_pool`), on who owns `OFT_env` (a cached singleton vs one per
worker), on masking (LCFS polygon alone, polygon plus a training-derived
valid mask, polygon plus the solver's native NaNs), and on size, from one
dominant module to a 27-module tree.

Prose, log and code are kept as three separate witnesses and never merged:
`prose_only_terms` is what the README claims over the code, `stated_vs_code`
is what the run's own log claims against it. `tools/judge_code.py` remains
the instrument for the qualitative call. Like `diagnostics`, nothing here
touches `contract.passed`.

```bash
python -m autotokamak.bench methodology --tag <tag>        # extract + print per run
python tools/aggregate_matrix.py --tag <tag> --show-rounds # cell table + every round
```

`aggregate_matrix.py` writes `methodology.csv` (one row per run: the chain)
and `methodology_rounds.csv` (one row per adaptive round: the logic) beside
`aggregate.csv`, and the HTML matrix report shows both per cell.

## Replicated campaigns

A cell run once is an anecdote: matrix-v1/v2 scored L3-pi at 0.0199 then
0.1350, and L2-claude_sdk at 0.189 then a crash. Campaigns now run each cell
n times (`bench run --rep N` records the index) via `tools/run_campaign.sh`,
and `tools/aggregate_matrix.py` groups by cell to report pass-rates with
Wilson intervals and rel-L2 medians with bootstrap CIs. Medians, not means —
one 0.96 cell makes a mean meaningless.

## reference_runs/

Archived agent-generated workspaces from the pre-refactor capability tests
(`L3-ursa/`, `L3-dspy/` — formerly top-level `just_ursa/` and `just_dspy/`).
The READMEs are tracked documentation of those experiments; the workspaces
are agent output, kept on disk, gitignored, and NEVER edited.

## Prompt versioning

Task problem texts are frozen comparability assets: **runs are only
comparable within one prompt version.** Any change to a problem text goes
into a new `_v2`/`_v3` file with a bumped `prompt_version:` field and a
header explaining what changed and why — never an in-place edit. Each run's
`result.json` records `task.prompt_version` and the trace records the YAML's
sha256, so every result is attributable to its exact prompt. Version changes
must be process-level (engineering-discipline gates, identical for every
harness) — never physics/ML hints, and never per-harness. The version ladder
is itself data: what each added gate does to where agents fail is part of
the experiment.

Current versions: v1 = original capability-test text; v2 (2026-08-10) adds
the STORAGE VALIDATION GATE and DELIVERABLE SELF-TEST after the URSA agent
stored all-NaN datasets as successes and shipped an untested predict.py;
v3 (2026-08-18, mini tasks only) fixes budget numbers left over from the
full-size tasks — v2 mini stated "3 rounds of 100" and "500 per round /
10-round cap" simultaneously, a ~17x solver-budget ambiguity. Use
`L2_mini_v3.yaml` / `L3_mini_v3.yaml` for any new mini campaign.

## Adding a harness

One adapter module in `src/autotokamak/harnesses/` implementing
`Harness.run(task, workspace, *, run_dir, model, timeout_seconds) → RunResult`
plus a registry entry in `harnesses/registry.py`. Smoke it with
`smoke.yaml` before any paid run.
