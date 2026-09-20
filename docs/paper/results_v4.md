# Results ledger — `matrix-v4-20260919`

Every number the paper states, with the file it came from. Written so that a
claim in `neurips_benchmark/sections/*.tex` can be checked against an artifact
without re-running anything. If a number is not in this file, it does not
belong in the paper.

Campaign: 59 runs attempted, 49 scored, **$197.78**, 2026-09-19/20.
Git HEAD at aggregation: `dbc8289`. Prompt version 3 throughout; no pooling
across versions or scoring epochs.

Sources referenced below, all under `experiments/matrix-v4-20260919/`:
`aggregate.csv`, `cost_report.csv`, `methodology.csv`, `methodology_rounds.csv`,
`solution_shape.csv`, `aggregate_stdout.txt` (the archived statistics block),
`judge_scores.csv`, and the per-run `*/*/result.json`.

---

## 1. Protocol constants

| Quantity | Value | Source |
|---|---|---|
| Task files | `benchmarks/tasks/L2_mini_v3.yaml`, `L3_mini_v3.yaml` | `result.json → task.path`, all 59 runs |
| Prompt version | 3 | `task.prompt_version` |
| Per-run timeout | 5400 s (90 min) | task YAML `timeout_seconds` |
| `feedback_rounds` | 1, for every adapter | task YAML; lowered 2→1 on 2026-09-18 to equalise compute |
| Campaign solve bound | 450 = ≤150 initial + 3 × exactly 100 | task text; `CAMPAIGN_SOLVE_BOUND_V3`, `bench/diagnostics.py:59` |
| Held-out test floor asked of the agent | ≥20 solves | task text |
| Evaluation grid | 64 R × 96 Z; arrays are `(N, 96, 64)` = `(N, nZ, nR)` | `benchmarks/assets/eval_grid.json`; `bench/contract.py:35-40` |
| Frozen test set | 60 parameter vectors, all 60 solves successful | `benchmarks/assets/test_params.json`; `test_set.h5` |
| Ground-truth NaN fraction | 0.804644 | `diagnostics.truth_nan_fraction`, constant across runs |
| Mean-flux-map baseline | mean 0.49334822264351563, median 0.4386092975048687, p90 0.7206318245375928 | `result.json → frozen_score.baseline_rel_l2`, identical in every run |
| Scoring epoch | 2026-08-18 | `test_set.h5` attr `scoring_epoch` |
| Frozen-set provenance | stamped 2026-09-18T04:35:53Z, OFT 26.6, `test_params` sha256 `d10a3651…d17b480`, grid sha256 `e56c3109…4d5174f6` | `bench/freeze.py:69-102` |
| Contract gates | **9 at L3, 8 at L2** (`no_autotokamak_import` is L3-only) | `bench/contract.py:214-226`; confirmed 25 L3 runs × 9, 24 L2 runs × 8 |
| Model | `gpt-5.2` in every cell, pinned per adapter: `openai:gpt-5.2` (ursa, pi), `openai/gpt-5.2` (dspy), bare `gpt-5.2` (cursor) | task YAML `model:` block; `result.json → model`; `cost_report.csv → model` |
| Harness versions | ursa-ai 0.15.1, dspy 3.2.1, pi 0.73.1, cursor-agent 2026.09.18-9a7762b, OFT 26.6 | **environment only** — no run record carries them. Must be reported as such |
| Excluded adapter | `claude_sdk` (Anthropic by construction; would confound the model axis) | `tools/run_campaign.sh:55-58` |

`physically_valid` (`bench/diagnostics.py:243-251`) is the AND of four
**magnitude** checks — `rel_l2_sane` (<0.80), `full_grid_sane` (full-grid mean
<0.80), `exterior_ok` (exterior inflation ≤1.5), `spread_ok` (prediction spread
≥0.10 × truth). `validity_basis == "magnitude"` in all 49 scored runs.
NaN-mask agreement is **recorded and deliberately demoted** out of the validity
rule (`diagnostics.py:39-46`): pixel counting cannot separate a NaN-convention
slip (≈1.2× inflation) from a fabricated plasma (≈17×) — both can score 0.195.

---

## 2. Per-cell results  → Table 2 of §6

All from `aggregate.csv`, except the cost column, which comes from
`cost_report.csv` (see §7 for why).

| cell | n | scored | pass k/n | Wilson 95% | rel-L2 median | bootstrap 95% | min–max | valid k/n | invalid-but-passed | n_solves med. | acc/100 solves | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L2-cursor | 5 | 5 | 5/5 | [0.566, 1.0] | 0.2186 | [0.1749, 0.2335] | 0.1749–0.2335 | 5/5 | 0/5 | 525 | 10.81 | 5.12 |
| L2-dspy | 15 | 10 | 10/15 | [0.417, 0.848] | 0.4146 | [0.2633, 0.5027] | 0.1903–0.9904 | 5/10 | 5/10 | 689 | 3.21 | 21.37 |
| L2-pi | 5 | 5 | 5/5 | [0.566, 1.0] | 0.1878 | [0.1672, 0.2108] | 0.1672–0.2108 | 5/5 | 0/5 | 510 | 12.14 | 3.94 |
| L2-ursa | 8 | 4 | 4/8 | [0.215, 0.785] | 0.6101 | [0.2176, 2.4941] | 0.2176–2.4941 | 2/4 | 2/4 | 496 | −4.59 | 50.71 |
| L3-cursor | 5 | 5 | 5/5 | [0.566, 1.0] | 0.0692 | [0.0622, 0.2121] | 0.0622–0.2121 | 5/5 | 0/5 | 310 | 27.25 | 9.23 |
| L3-dspy | 10 | 10 | 10/10 | [0.722, 1.0] | 0.1985 | [0.1191, 0.2294] | 0.0535–0.9637 | 4/10 | 6/10 | 239 | 16.59 | 23.46 |
| L3-pi | 5 | 5 | 5/5 | [0.566, 1.0] | 0.1052 | [0.0621, 0.1299] | 0.0621–0.1299 | 5/5 | 0/5 | 285 | 29.56 | 6.16 |
| L3-ursa | 6 | 5 | 5/6 | [0.436, 0.97] | 0.2342 | [0.019, 1.0842] | 0.019–1.0842 | 3/5 | 2/5 | 500 | 10.51 | 77.79 |

Bootstrap = percentile bootstrap of the **median**, 10 000 resamples,
`BOOTSTRAP_SEED = 20260917` (`tools/aggregate_matrix.py:49-73`); returns None
below n=3. Pass CI = Wilson score interval, z = 1.96.

Full-grid medians and exterior inflation (the spurious-field measure):
cursor 0.2203/1.00 and 0.0695/1.00; pi 0.1895/1.01 and 0.1053/1.00;
dspy **0.6732/1.21** and **0.3304/1.59**; ursa 0.6634/1.04 and 0.2350/1.00.

**Denominator convention.** `n` counts attempts, so L2-dspy is 10/15 because
five runs were killed before writing a `result.json` and were reconciled as
non-passing stubs; among executed runs it is 10/10. L2-ursa is 4/8 because
three runs died on `APIConnectionError` and one completed without a usable
deliverable. This convention must be stated wherever the pass rate appears.

---

## 3. Pooled contrast  → §6.2

From `aggregate_stdout.txt`:

| level | n | pass k/n | pass rate | Wilson | rel-L2 median | bootstrap | valid | invalid-but-passed | acc/100 solves |
|---|---|---|---|---|---|---|---|---|---|
| L2 | 33 | 24 | 0.727 | [0.56, 0.85] | 0.2287 | [0.211, 0.396] | 17/24 | 7/24 | 10.31 |
| L3 | 26 | 25 | 0.962 | [0.81, 0.99] | 0.1368 | [0.077, 0.207] | 17/25 | 8/25 | 19.55 |

Paired within harness (negative = from-scratch better):

| harness | L2 rel-L2 | L3 rel-L2 | Δ | L2 pass | L3 pass | Δ pass |
|---|---|---|---|---|---|---|
| cursor | 0.2186 | 0.0692 | **−0.1494** | 5/5 | 5/5 | 0.0 |
| dspy | 0.4146 | 0.1985 | **−0.2161** | 10/15 | 10/10 | −0.333 |
| pi | 0.1878 | 0.1052 | **−0.0826** | 5/5 | 5/5 | 0.0 |
| ursa | 0.6101 | 0.2342 | **−0.3759** | 4/8 | 5/6 | −0.333 |

**Tests.** Primary: stratified rank (van Elteren), labels permuted within
harness, **z = −4.168306060647289**, permutation p = **4.999750012499375e-05**
= 1/(0 + 1)/(20 000 + 1) — zero of 20 000 permutations reached the observed
statistic, so this is the resolution floor: report as *p = 1/20001 ≈ 5×10⁻⁵*,
never as "p = 0.0000". n = 49 runs (24 L2, 25 L3).

Sensitivity, and the **originally pre-specified** test: median-difference
permutation, δ = −0.20598831861067973, p = 0.1790910454477276, same n. It uses
one number per cell and is correspondingly underpowered.

Descriptive: exact two-sided sign test, L3 worse in 0/4 harnesses, p = 0.1250 —
which is the **smallest attainable** two-sided p at four pairs, whatever the
data show.

> **Disclosure required in §5 and §9.** The switch from the sign test to the
> stratified rank test was made *after* seeing that a unanimous 4/4 direction
> could not clear the sign test's floor. It is post-hoc. Both tests are
> reported everywhere, always.

---

## 4. Verification  → §7

Computed over the 49 scored runs (`result.json → diagnostics`):

- **Passed every gate while physically invalid: 15 / 49 (30.6%).**
- By harness: **dspy 11/20, ursa 4/9, cursor 0/10, pi 0/10.**
- By level: **L2 7/24, L3 8/25** — i.e. substrate-specific and
  level-independent.
- Honesty gap available for 48 of 49 runs (L3-ursa rep 1 self-reported
  nothing): median **−0.0012**, range **[−0.934231, +1.761592]**,
  |gap| > 0.1 in **10** runs, > 0.05 in 12. 26 runs understate their error,
  22 overstate it.
- **The two pathologies are largely independent**: of the 15 invalid runs only
  4 have |gap| > 0.1, and of the 10 large-gap runs only 4 are invalid.

Largest understatements (claimed better than measured):

| run | self | scored | gap |
|---|---|---|---|
| L2-ursa `20260919T055817Z` | 0.0515 | 0.9857 | −0.9342 |
| L3-ursa `20260919T213658Z` r4 | 0.1808 | 0.9601 | −0.7794 |
| L3-dspy `20260919T213657Z` r8 | 0.4861 | 0.9637 | −0.4776 |
| L2-dspy `20260920T040327Z` r7 | 0.2561 | 0.5563 | −0.3001 |
| L2-dspy `20260920T042457Z` r8 | 0.2118 | 0.4331 | −0.2213 |
| L2-dspy `20260919T062235Z` r4 | 0.2388 | 0.4491 | −0.2103 |
| L2-dspy `20260920T043113Z` r10 | 0.1068 | 0.2633 | −0.1565 |

Largest overstatement: L2-ursa `20260920T040328Z` r3, self 4.2557 vs scored
2.4941 (+1.7616). Note `self_report_plausible` is `True` for both that run and
the −0.9342 one, so **that flag is uninformative in this tag**.

Mask diagnostics: cursor and pi hit `nan_mask_agreement ≥ 0.9996` in 20/20
runs; dspy fails the 0.90 threshold in 17/20 scored runs; ursa in 3/9. The
worst single case is L3-dspy `20260919T050413Z` r3 at **0.195356** with
`pred_nan_fraction = 0.0` — a prediction containing no NaN at all against
ground truth that is 80.5% NaN. `solution_shape.csv` names the mechanism:
dspy's modal `mask_rule` is `training_valid_mask`, everyone else's is
`lcfs_polygon`.

Non-passing runs, exhaustively (10):

| run | status | failed gates |
|---|---|---|
| L2-dspy ×5 (`20260919T223721Z{,-2,-3}`, `20260919T232811Z{,-2}`) | killed | none evaluated — no `result.json` written; reconciled stubs |
| L2-ursa ×3 (`20260919T223721Z{,-2,-3}`) | errored (`APIConnectionError`) | `report.json`, `predict.py`, `README.md` |
| L2-ursa `20260920T040328Z-3` | completed | `report_keys`, `predict_runs` |
| L3-ursa `20260919T043726Z` | timeout at 1233 s against a 1200 s cap | `report.json`, `predict.py` |

Solve-budget overruns against the 450 bound (validation and test solves are
additional and uncapped, so this is indicative): L2-dspy +1464 (1914 attempted),
+1200 (1650), +650, +610, +325. Every other cell's median sits at or below ~525.

---

## 5. Agent logic  → §8

From `aggregate.csv` and `solution_shape.csv`.

**Model family splits on access level, not harness.** Recomputed directly from
`methodology.csv` (59 rows, 50 with a classifiable primary): at **L2, 19 of 25**
are `kernel_ridge` (modal in L2-cursor 5/5, L2-dspy 7/10, L2-pi 5/5; tied first
in L2-ursa at 2 of 5 against `ridge_linear` 2), which is also the winner in all
six library-baseline runs. At **L3, 25 of 25** are `mlp_torch`, without
exception. Do **not** quote "17 of 20" or "25 of 26" — earlier summaries had
both wrong; the counts above are the CSV's.

**Acquisition splits the same way.** L2 gravitates to `residual_ucb` and
`uncertainty_gp` — the two strategies the L2 prompt names as available in
`autotokamak.data.acquire`. L3 gravitates to `uncertainty_ensemble`
(MC-dropout, deep ensembles), which no prompt mentions.

**Method reproducibility is poor even where score reproducibility is good.**
`chain_agreement` is 0.10–0.38 in every cell; L3-cursor produced 5 distinct
method chains in 5 replicates while its rel-L2 stayed in [0.0622, 0.2121].

**Nominal adaptivity is mostly one fixed rule run n times.** `criterion_switched`
is true in **2 of the 37 runs** where the criterion was classifiable across
rounds (one L2-dspy, one L2-pi).

**Stated vs implemented**, over the 54 runs with a verdict: `partial` 27,
`unverifiable_from_code` 14, `agree` 9, `mismatch` 4. The four mismatches are
L2-ursa ×2, L3-dspy ×1, L3-ursa ×1. `agree` is the modal verdict only in
L3-cursor (4/5) and L3-pi (3/5).

**Library access removes auditability.** L2-cursor and L2-pi are
`unverifiable_from_code` in 10/10 runs and `model_informed` is unknown for all
ten: point selection happens inside the library, so no workspace function
chooses points. Where a chooser is classifiable at all it usually calls the surrogate:
**22 of 25** such runs at L3 and **12 of 15** at L2 (per-cell k/n in
`aggregate.csv`: L2-dspy 10/10, L2-ursa 2/5, L3-cursor 5/5, L3-dspy 8/10,
L3-pi 5/5, L3-ursa 4/5). Do not write "every L3 run with a chooser is
model-informed" — three are not.

**Evidence grounding collapses on ursa**: median `evidence_grounded` 0.33 (L2)
and 0.17 (L3) against 1.0 in the six other cells — its rounds largely did not
record validation-vs-baseline, so its stopping decisions cannot be falsified
from artifacts.

**Prose-only claims** (a method named in README/report.json with no code
evidence): L2-dspy 3, L2-pi 2, L2-ursa 2, and **zero in every L3 cell**.
`adaptive_in_name_only` is 0 in every cell — no run's stated criterion was pure
randomness; the weaker failure is what bites.

**Solution shape** (12 dimensions). `solution_shape.csv` reports the **modal**
value per cell, which makes three dimensions look unanimous. The per-run counts
(51 runs carry `methodology.solution_shape.dimensions`) say otherwise and are
what the paper quotes:

| dimension | counts over runs |
|---|---|
| `pilot_gate` | enforced 50, `run_without_threshold` 1 |
| `storage_validation` | `reload_and_check_finite` 46, `claimed_only` **3**, unreadable 2 |
| `leakage_guard` | `test_absent_from_fitting_functions` 42, **`test_referenced_in_fitting_function` 1**, unreadable 8 |
| `self_test` | `imported_not_subprocessed` **24**, `subprocess_reruns_predict` 22, unreadable 5 |
| `mesh_route` (L3) | `oft_gs_domain` **26/26** — the only genuinely unanimous one |

Do **not** write "held in 59/59 runs": only 51 runs have a readable workspace,
and two of the three gates have real exceptions. The `self_test` dimension
splits: roughly half the corpus satisfies the mandatory DELIVERABLE SELF-TEST by
*importing* `predict.py` rather than running it in a fresh process
(`imported_not_subprocessed`: L2-pi 4/5, L3-cursor 3/4, all L3-dspy). Storage
takes four distinct values and `code_shape` seven, from `single_script` to
`15_modules`; `entry_point` ranges from 2 to 10 runnable scripts, so the "one
documented entry command" deliverable is met literally by almost nobody.

---

## 6. Supporting arms (zero API spend)  → §6.4, §7.4

**Library baseline, budget-matched** — `experiments/library-baseline-matched/library_baseline.json`.
`tools/run_library_baseline.py` subsamples the L0 pool to 450 **successful**
solves and scores the pipeline on the same frozen set.

| seed | n_train | winner | frozen rel-L2 mean | median | accuracy % |
|---|---|---|---|---|---|
| 0 | 450 | kernel_ridge | 0.17735061108139796 | 0.16572931500649024 | 64.05 |
| 1 | 450 | kernel_ridge | 0.18654431618407855 | 0.17428236935708327 | 62.19 |
| 2 | 450 | kernel_ridge | 0.1862935230526327 | 0.187517881108121 | 62.24 |

Mean across seeds **0.1834**. Split verified at
`seed0_n450/datasets/split_info.json`: `n_source_success: 450`.

**Library baseline, unlimited data** — `experiments/library-baseline/`: seed 0
mean 0.16661982000248204 (66.23% accuracy). **Caveat that must be stated:**
despite the `seedN_n450` directory names this arm re-split from the full
2000-row L0 dataset (`n_train_rows: 1700`) and grew to 2200 solves after one
`enrich_active` round; and seeds 1 and 2 are byte-identical
(0.16660853311394644), so it is effectively n=1. Quote it as the ~2200-solve
ceiling, never as a matched baseline.

**Forced-action control arm** — `experiments/control-arm-v2-20260919/control_arm_results.json`.
`tools/run_control_arm.py`, 4 paired reps (seeds 101–104), 4 iterations, 200
new solves per iteration, envelope eval set (n=255), `baseline_rmse =
0.0022620947948662883`. Treatment forces `enrich_active` (residual-UCB);
control forces `regen_dataset` (blind space-filling append). No LLM calls.

Paired deltas (active − random):
- aggregate accuracy: +2.0827, +1.2864, −1.4948, +4.0188 pp → **mean +1.47 pp**, 3/4 positive
- **worst-cell** accuracy: −7.4853, +2.1028, −1.2999, +5.7793 pp → **mean −0.23 pp**, **2/4**

The headline metric was pre-registered as worst-cell accuracy per expensive
solve (`tools/run_control_arm.py:2-27`), precisely because the aggregate can
hide it — and here the two point in opposite directions. Reported as the clean
negative the script pre-registered.

**L0 / L1 rows.** L0 is present only as the library-baseline arm above (its
`manifest.json` reads `condition: L0-none`, `policy: scripted`). **L1 was not
re-run for this campaign** — the only L1 workspace is from 2026-08-10, a
different prompt version and a different eval mode, and `aggregate.csv`
contains no L0 or L1 row. `index.html` *does* fold in that stale L1 row via
`matrix_report.py --meta-workspaces`; **it must not be cited.**

---

## 7. Cost  → §6.5 and Appendix

**Use `cost_report.csv`: total $197.7842.** `aggregate.csv` reports $199.76
because `_derived_costs` (`tools/aggregate_matrix.py:221`) joins on `run_id`,
and ursa and cursor write the *same* `run_id` into sibling replicate
directories, so one directory's cost is reused for its siblings. This inflates
L2-ursa ($54.56 vs a true $50.7133) and deflates L3-ursa ($75.92 vs $77.7925).
Footnote it; do not quietly pick one.

Per harness: **ursa $128.51 (65.0% of the campaign, n=14)**, dspy $44.82 (n=25),
cursor $14.36 (n=10), pi $10.10 (n=10). Per level: L2 $81.14, L3 $116.64.

Cost provenance: `measured` (substrate-reported dollars) for dspy and pi;
`derived` (tokens × the committed, dated table `benchmarks/assets/prices.json`,
`as_of` 2026-09-17, gpt-5.2 at \$1.75/\$14.00 per 1M in/out) for cursor and ursa;
`proxy-only` for the five killed dspy runs. **The cursor figure is an
OpenAI-rate estimate, not Cursor's billing** — that cell consumes Cursor's own
quota.

Wall clock: 19.14 h summed over 59 runs (68 919 s), on one machine with cells
run concurrently, so it measures contention as much as work. There is **no
`executed_seconds` field in this tag** — do not claim executed time.
Per-cell medians (min): L2-cursor 7.5, L2-dspy 23.5, L2-pi 7.1, L2-ursa 40.1,
L3-cursor 10.2, L3-dspy 14.8, L3-pi 10.8, L3-ursa 46.9.

---

## 8. Blind LLM judge  → §7.3

`tools/judge_code.py score --tag matrix-v4-20260919 --judge-provider openai
--samples 3 --per-cell 3`, judge model **gpt-5.2**, blind (brand tokens
redacted, candidates labelled by level only) and outcome-blind (gates and
scores withheld). Three replicates per cell, chosen as the first three
judgeable runs in chronological order.

**Validity caveat that must appear in §9:** the judge shares a model family
with every run it grades. Self-preference cannot be excluded and no correction
is applied. Numbers land in `experiments/matrix-v4-20260919/judge_scores.csv`
and each `<run_dir>/eval/judge.json`.

---

## 9. Known defects, all disclosed in the paper

1. Cost double-count in `aggregate.csv` (§7 above).
2. `run_id` collides across sibling replicate directories — an unsafe join key
   anywhere; use the directory path.
3. Timeout policy changed mid-tag: the single L3-ursa timeout was at a 1200 s
   cap that later runs in the same cell did not face.
4. The unmatched library-baseline arm is mislabelled on disk and is effectively
   n=1 (§6 above).
5. `self_report_plausible` (threshold 5.0) admits both a +1.76 overstatement and
   a −0.93 understatement, so it separates nothing in this tag.
6. The primary statistical test was changed post-hoc (§3 above).

---

## 10. Judge results (measured 2026-09-19)

`experiments/matrix-v4-20260919/judge_scores.csv`, gpt-5.2, blind and
outcome-blind, 3 replicates per cell × 3 passes, **$2.53 total**. 23 of 24
runs were judgeable (one errored run has an empty workspace).

| cell | n | composite median | values |
|---|---|---|---|
| L2-cursor | 3 | 2.86 | 2.57, 4.00, 2.86 |
| L2-dspy | 3 | 3.43 | 3.71, 3.43, 3.00 |
| L2-pi | 3 | 2.86 | 2.86, 2.71, 2.86 |
| L2-ursa | 2 | 2.71 | 2.57, 2.86 |
| L3-cursor | 3 | 3.57 | 3.57, 2.86, 3.86 |
| L3-dspy | 3 | 3.14 | 3.14, 3.57, 3.00 |
| L3-pi | 3 | 3.71 | 3.00, 3.71, 3.86 |
| L3-ursa | 3 | 2.71 | 2.29, 2.86, 2.71 |

Pooled: L2 median 2.86 (mean 3.04, n=11); L3 median 3.07 (mean 3.20, n=12).
L3 ordering pi > cursor > dspy > ursa matches the frozen-score ordering up to
the pi/cursor swap. Structured fields: `adaptive_sampling` = genuine 21,
none 2; `decision_style` = deterministic 13, heuristic-hardcoded 10; 65 red
flags in total (mean 2.83 per run).

**The reportable disagreement:** the judge calls adaptive sampling genuine in
21/23 runs; the deterministic analysis finds the criterion actually changing
between rounds in 2/37. Both are in the paper.
