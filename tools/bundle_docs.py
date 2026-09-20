# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""The prose that ships with a results bundle: a README and a data dictionary.

Kept apart from `make_results_bundle.py` because it is almost entirely text,
and because the column definitions are the part most likely to need editing
when the aggregation changes.

Nothing here is generated from the code it documents -- the CSV column names
are hand-written, because the meaning of `acc_per_100_solves` lives in an
author's head and not in a docstring. The cell *values* are a different matter
and are generated; see `write_glossary` in the packager.
"""
from __future__ import annotations

import csv
from pathlib import Path


def write_readme(dest: Path, tag: str, n_runs: int, n_traces: int,
                 paper: str | None, missing: list[str]) -> None:
    trace_line = (
        f"- `trace.json` — the agent's full turn-by-turn transcript: every prompt\n"
        f"  it was given and everything it said or ran. {n_traces} of them. This is\n"
        f"  the richest material here for a follow-up study, and the only part that\n"
        f"  reads like a narrative.\n"
        if n_traces else
        "- Agent transcripts were excluded from this bundle. Re-run the packager\n"
        "  without `--no-traces` to include them.\n")

    (dest / "README.md").write_text(f"""# autotokamak — campaign results, `{tag}`

A self-contained copy of the results from the agent benchmark campaign. Nothing
here needs the repository, an internet connection, or a Python environment: the
report opens in a browser and the tables open in anything.

**Start with `index.html`.** It is one file, about 10 MB, and it contains the
whole campaign: a matrix of every experimental cell, plain-English descriptions
of what each condition means, and for each of the 59 runs a detail block giving
what the agent was asked, what it says it did, what its code actually does, and
a true/predicted/error picture of the magnetic field it produced. Technical
terms in it are hover-annotated.

## What the experiment was

Fifty-nine coding agents were each given the same task: learn to drive a
Grad–Shafranov plasma-equilibrium solver, spend a fixed budget of at most 450
solver calls deciding which equilibria to compute, train a surrogate model
mapping five plasma-shape parameters to a 96×64 magnetic flux field, and hand
over a predictor. Every one was then scored by us, identically, on 60 held-out
cases none of them ever saw.

Two things were varied:

- **Access level.** `L2` agents could import a proven pipeline library. `L3`
  agents had to write everything, and the prohibition was enforced by static
  analysis of every file they wrote.
- **Agent framework** (we call it the *harness* or *substrate*): `ursa`,
  `dspy`, `pi`, `cursor`.

Everything else was held fixed — same task text, same model (`gpt-5.2`), same
budget, same scoring. `L0` in the tables is the reference point: the same
pipeline run by a scripted policy with no language model at all.

## The three findings

1. **From-scratch beat library-assisted in all four frameworks** (pooled median
   relative-L2 0.137 against 0.229). The library is not the problem — it beats
   five of the eight agent cells on its own. Given the library, all four
   frameworks adopt its defaults; given nothing, all four find something else.
2. **The framework mattered more than the scaffolding**, with the model held
   fixed: a factor of 8.8 across frameworks against 1.7 across access levels.
3. **15 of 49 scored runs passed every machine-checkable gate while being
   physically invalid.** The clearest case scored better than most of the
   corpus while emitting a magnetic field throughout the vacuum region, where
   there is no field. See Figure 7 of the paper, or that run's panel in
   `index.html`.

## Three caveats that must travel with this data

If you build on these numbers, these need saying alongside them.

1. **Denominators count attempts, not survivors.** `L2-dspy` is 10/15 because
   five runs were killed by the driver before they could write anything; among
   runs that executed it is 10/10. Both conventions are defensible and the
   choice moves the pooled `L2` pass rate materially.
2. **The primary statistical test was chosen after seeing the data.** We
   pre-specified a sign test over four paired substrates, found it cannot
   return p < 0.125 at any effect size, and moved to a stratified rank test.
   All three tests agree in direction; the p-value depends on the choice. All
   three are printed in `aggregate_stdout.txt`.
3. **Two cost figures exist and they differ.** `cost_report.csv` sums to
   $197.78 and is the one to use. `aggregate.csv` reports $199.76 because it
   joins costs on a run identifier that two frameworks reuse across sibling
   replicate directories.

Two smaller ones: the `library-baseline` (unmatched) arm is mislabelled on disk
— its directories say `n450` but it trained on the full 2000-row pool, and two
of its three seeds are bit-identical, so treat it as n=1 at ~2200 solves. And
one `L3-ursa` run was killed at a 1200 s cap that later runs in the same cell
did not face.

## What is in here

### Top level — the aggregated results
{chr(10).join(f"- `{name}` — {desc}" for name, desc in _TAG_FILES)}

`DATA_DICTIONARY.md` defines every column of every CSV. `GLOSSARY.md` defines
every canonical *value* those columns can take (`lcfs_polygon`,
`residual_ucb`, `stop_without_threshold`, and so on).

### `runs/<cell>/<run_id>/` — {n_runs} runs, one directory each
- `result.json` — the machine-readable record: harness outcome, every contract
  gate with its verdict, the frozen-set score, the physical-validity
  diagnostics, and the extracted methodology.
{trace_line}- `judge.json` — the blind code review, for the 24 runs sampled for it.

### `supporting_arms/` — the three zero-cost control experiments
- the library baseline at a matched budget and unlimited data
- the forced-action control arm testing whether adaptive sampling pays

### `paper/`
{"- `paper.pdf` — the write-up, built from `" + paper + "`" if paper else "- (no built paper was found at packaging time)"}
- `figures/` — its figures as vector PDFs

## What is deliberately not here

The agent workspaces (3.2 GB of training datasets and model pickles) and the
raw harness event streams (6.9 GB). Both live in the repository. If you want to
re-run an agent's predictor rather than read about it, you need the workspace,
and the repository README explains how.

Note that agent-written files inside those workspaces sometimes contain
absolute paths from the machine the campaign ran on. That is agent-generated
output and was deliberately not edited.
{"" if not missing else chr(10) + "Missing from this bundle (absent from the campaign directory): " + ", ".join(missing) + "." + chr(10)}
## Reproducing or extending

The repository is at `https://github.com/hindyros/autotokamak`. Its README
covers installing, running the benchmark on a new agent, and rebuilding every
table and figure in the paper from these artifacts.
""")


def write_data_dictionary(dest: Path, tag_dir: Path) -> None:
    """Define every column of every CSV that ships in the bundle."""
    parts = [
        "# Data dictionary",
        "",
        "Every column of every table in this bundle. Definitions are written out",
        "rather than generated, because what a column *means* is not recoverable",
        "from the code that writes it.",
        "",
        "Throughout: **rel-L2** is the per-case relative L2 error of a predicted",
        "flux field against the truth, computed only where the truth is defined.",
        "The **mean-map baseline** is what you score by ignoring the input and",
        "always predicting the average field: 0.4933 on this test set. **Accuracy**",
        "is `100 x (1 - rel_l2 / 0.4933)`, so 0% is 'no better than not trying'.",
        "",
    ]
    for name, cols in _COLUMNS.items():
        path = tag_dir / name
        shape = ""
        if path.is_file():
            with path.open() as fh:
                n = sum(1 for _ in csv.reader(fh)) - 1
            shape = f" — {n} data rows"
        parts += [f"## `{name}`{shape}", "", _TABLE_INTRO[name], ""]
        parts += ["| column | meaning |", "|---|---|"]
        parts += [f"| `{c}` | {d} |" for c, d in cols]
        parts += [""]
    parts += _CLOSING
    (dest / "DATA_DICTIONARY.md").write_text("\n".join(parts))


# --- the tables, described ---------------------------------------------------

_TAG_FILES = [
    ("index.html",             "the browsable report — **start here**"),
    ("aggregate.csv",          "one row per experimental cell; the paper's main table"),
    ("methodology.csv",        "one row per run: the chain of methods it chose"),
    ("methodology_rounds.csv", "one row per adaptive round: the decision logic"),
    ("solution_shape.csv",     "12 design questions × 8 cells, from static analysis"),
    ("cost_report.csv",        "one row per run: dollars, tokens, wall time, solves"),
    ("judge_scores.csv",       "blind code-review rubric scores"),
    ("aggregate_stdout.txt",   "the statistics: pooled contrast and all three tests"),
    ("judge_score_stdout.txt", "console output of the code review"),
]

_TABLE_INTRO = {
    "aggregate.csv":
        "One row per experimental cell (access level × agent framework). This is "
        "the table behind the paper's main results.",
    "methodology.csv":
        "One row per run. Produced by deterministic static analysis of the code "
        "each agent wrote — no language model is involved, so it re-extracts "
        "identically. Columns come in two families: what the agent *said* it did "
        "(from its README and report) and what its *code* actually computes.",
    "methodology_rounds.csv":
        "One row per adaptive sampling round of every run: what the agent chose, "
        "why it said it chose it, and whether the evidence for that choice was "
        "recorded at the time.",
    "solution_shape.csv":
        "Twelve questions about how each run solved the whole task, not just the "
        "acquisition step. Rows are dimensions, columns are cells. Each entry is "
        "the cell's *modal* value with the number of replicates agreeing; a modal "
        "value can hide a minority, so per-run values are in each `result.json`.",
    "cost_report.csv":
        "One row per run directory. Note that two frameworks reuse a run "
        "identifier across sibling replicate directories, so join on the "
        "directory path and not on `run_id`.",
    "judge_scores.csv":
        "A blind, outcome-blind rubric review of the code, three replicates per "
        "cell. The reviewer never saw which framework wrote the code, nor its "
        "score. It is the same model family that produced the runs, so treat "
        "these as corroboration and not as an independent measurement.",
}

_COLUMNS = {
    "aggregate.csv": [
        ("condition", "the cell, `<level>-<harness>`"),
        ("level", "`L2` (may import the library) or `L3` (from scratch, audited)"),
        ("harness", "the agent framework: ursa, dspy, pi or cursor"),
        ("n", "runs **attempted** in this cell"),
        ("n_completed", "runs whose harness reported completion"),
        ("n_timeout", "runs killed by a timeout"),
        ("n_errored", "runs that ended in an error"),
        ("pass_k", "runs passing every contract gate"),
        ("pass_rate", "`pass_k / n` — note the denominator is attempts"),
        ("pass_ci_lo, pass_ci_hi", "Wilson score interval on the pass rate, z = 1.96"),
        ("n_scored", "runs that produced a usable predictor and so have a score"),
        ("rel_l2_median", "median rel-L2 over scored runs. **The headline number.**"),
        ("rel_l2_ci_lo, rel_l2_ci_hi", "percentile bootstrap interval on that median, 10 000 resamples, seed 20260917; empty below n = 3"),
        ("rel_l2_min, rel_l2_max", "best and worst scored run in the cell"),
        ("rel_l2_full_median", "the same metric recomputed with the plasma's *exterior* charged to the numerator. Compare with `rel_l2_median`: a large gap means the predictor invented a field where none exists"),
        ("ext_inflation_median", "ratio of the two. ~1.0 clean, ~1.2 a NaN-convention slip, ~17 a fabricated plasma"),
        ("n_solves_median", "median solver calls attempted. The task's campaign bound was 450, plus uncapped validation and test solves"),
        ("acc_per_100_solves", "accuracy divided by (solves/100): a crude efficiency measure. Negative where the cell median is worse than the mean map"),
        ("phys_valid_k, phys_valid_n", "runs passing all four physical-validity checks, of those assessable"),
        ("passed_but_invalid_k, passed_but_invalid_n", "**runs that passed every gate and were physically invalid**, of those assessable. The paper's central measurement"),
        ("honesty_gap_median", "median of (self-reported rel-L2 − measured rel-L2). Negative means the agent claimed to be better than it was"),
        ("cost_usd_mean, cost_usd_total", "model spend. Derived from tokens for cursor and ursa; see the README caveat about the total"),
        ("n_with_chain", "runs whose method chain could be extracted"),
        ("chain_agreement", "share of replicates on the cell's modal chain — *method* reproducibility, which is not score reproducibility"),
        ("n_distinct_chains", "how many different pipelines the cell's replicates built"),
        ("chain_modal", "the modal chain, as `design -> representation+model[ensembling] -> acquisition x rounds -> stop`"),
        ("model_primary", "which model families the runs settled on, with counts"),
        ("acq_classes", "acquisition strategies named, with counts"),
        ("rounds_median", "median number of adaptive rounds actually run"),
        ("criterion_switched_k, criterion_switched_n", "replicates whose acquisition criterion *changed* between rounds — i.e. genuinely reactive adaptivity — of those where it could be classified"),
        ("evidence_grounded_median", "share of rounds whose validation-vs-baseline error was recorded at the moment of choosing. Low values mean the stopping decision cannot be checked"),
        ("stop_decisions", "why runs stopped: threshold met, round cap, or unexplained"),
        ("code_acq", "acquisition strategies found in the *code*, with counts"),
        ("code_logic_modal", "modal implemented logic, as `acq:...|sel:...|model_informed:yes/no`"),
        ("code_logic_agreement, n_distinct_code_logics", "agreement and variety of the implemented logic"),
        ("selection_rule", "how candidates are picked once scored: top-k, threshold, random draw"),
        ("model_informed_k, model_informed_n", "replicates whose point-choosing function actually calls the surrogate, of those with a classifiable chooser"),
        ("stated_vs_code", "verdict of comparing stated against implemented criterion: agree, partial, mismatch, or unverifiable"),
        ("claimed_not_implemented", "criterion families named in prose that the code does not compute"),
        ("adaptive_in_name_only_k", "replicates whose stated criterion is pure randomness (zero everywhere in this campaign)"),
        ("prose_only_claim_k", "methods named in the README or report with no code evidence at all"),
    ],
    "methodology.csv": [
        ("tag, condition, level, harness, run_id, replicate", "which run this is. Join on `condition` + `run_id`, or better on the directory path"),
        ("status", "completed, killed, errored or timeout"),
        ("chain", "the full chain of methods this run built"),
        ("model_primary", "its primary model family"),
        ("acq_classes", "acquisition strategies it *named*"),
        ("n_rounds", "adaptive rounds actually executed"),
        ("criterion_switched", "did the criterion change between rounds"),
        ("evidence_grounded", "fraction of rounds with recorded validation evidence"),
        ("stop_decision", "`stop_threshold_met`, `stop_without_threshold` or `stop_unexplained`"),
        ("adaptive_in_name_only", "stated criterion was pure randomness"),
        ("prose_only", "named a method with no code evidence"),
        ("code_acq", "acquisition strategies found in the code"),
        ("code_logic_signature", "compact summary of the implemented selection logic"),
        ("code_evidence_scope", "whether the evidence was resolved to a function or only a file"),
        ("selection_rule", "top_k, threshold or random_draw"),
        ("model_informed", "does the chooser actually call the surrogate"),
        ("stated_vs_code", "agree / partial / mismatch / unverifiable_from_code"),
        ("only_stated, only_implemented", "criterion families on one side but not the other"),
        ("chooser_where", "`file:line` of the functions that choose points"),
        ("shape_signature", "compact summary of the 12 solution-shape dimensions"),
        ("rel_l2", "the run's measured error"),
        ("contract_passed", "did it pass every gate"),
        ("physically_valid", "did it pass all four physical checks"),
    ],
    "methodology_rounds.csv": [
        ("tag, condition, run_id, replicate", "which run"),
        ("round", "1-indexed adaptive round"),
        ("n_acquired", "new solver calls this round"),
        ("criterion_classes", "the criterion, classified into the shared vocabulary"),
        ("criterion_source", "where the criterion was read from"),
        ("criterion_text", "the agent's own words, verbatim"),
        ("val_rel_l2, baseline_rel_l2", "validation error and its baseline at this round"),
        ("val_over_baseline", "their ratio — the task's stopping rule is ≤ 0.30"),
        ("met_stop_threshold", "did that ratio clear the task's threshold"),
        ("evidence_grounded", "was the above actually recorded, or inferred"),
        ("decision", "what the run did next"),
        ("reason_coverage", "how much of the stated reason could be classified"),
    ],
    "solution_shape.csv": [
        ("dimension", "which design question this row answers"),
        ("question", "the question in words"),
        ("L2-cursor, L2-dspy, L2-pi, L2-ursa, L3-cursor, L3-dspy, L3-pi, L3-ursa", "that cell's modal answer, with `(k/n)` replicates agreeing where not unanimous. A dash means no workspace code answers it — at L2 this usually means the library owns that part"),
        ("n_distinct", "how many different answers appeared across the eight cells"),
        ("agreement", "`all_same`, `mixed` or `all_differ`"),
    ],
    "cost_report.csv": [
        ("condition, run_id", "which run — but see the README: `run_id` is not unique"),
        ("status", "completed, killed, errored, timeout"),
        ("model", "the model string, as recorded by the adapter"),
        ("wall_min", "wall-clock minutes. Contaminated by concurrency; not an efficiency metric"),
        ("turns", "agent turns, where the substrate reports them"),
        ("tok_in, tok_out, tok_cache", "token counts"),
        ("cost_usd", "dollars"),
        ("cost_source", "`measured` (the substrate reported dollars), `derived` (tokens × a dated price table), or `proxy-only` (no figure available)"),
        ("n_solves", "solver calls attempted. One row holds a breakdown dict rather than an integer"),
        ("rel_l2_mean", "the run's measured error"),
    ],
    "judge_scores.csv": [
        ("condition, run_id", "which run"),
        ("correctness, methodology, structure, robustness, reproducibility, efficiency, documentation", "the seven rubric dimensions, 1–5, median of three passes"),
        ("composite", "their mean"),
        ("decision_style", "the reviewer's classification of how decisions were made"),
        ("adaptive_sampling", "whether the reviewer judged the adaptivity genuine. Compare against `criterion_switched` in `methodology.csv`: the reviewer says genuine in 21 of 23, the parser finds it reactive in 2 of 37"),
        ("n_red_flags", "concerns the reviewer raised"),
        ("judge_cost_usd", "cost of reviewing this run"),
    ],
}

_CLOSING = [
    "## Per-run records",
    "",
    "`runs/<cell>/<run_id>/result.json` carries more than the CSVs summarise.",
    "The blocks worth knowing about:",
    "",
    "- `contract.gates` — every gate with its verdict, and evidence for failures.",
    "- `frozen_score` — the scored error, the baseline, accuracy, and the",
    "  prediction-shape statistics: `nan_mask_agreement` (does the prediction's",
    "  undefined region match the truth's), `pred_nan_fraction` against",
    "  `truth_nan_fraction` (0.8046 on this test set), `exterior_inflation`, and",
    "  `pred_over_truth_spread` (below 0.1 means a near-constant predictor).",
    "- `diagnostics` — the four physical-validity checks with their thresholds,",
    "  `passed_gates_but_invalid`, the honesty gap, and the solve accounting.",
    "- `methodology` — the extracted method chain, the per-round decision logic,",
    "  and `solution_shape.dimensions`, where every answer carries `file:line`",
    "  evidence pointing into the agent's own code.",
    "",
    "`trace.json` is the agent's transcript. Its shape differs by framework,",
    "since each adapter maps its substrate's native events onto a shared",
    "round/step structure, but `prompt` (the task given) and the round/step",
    "sequence are common to all.",
]
