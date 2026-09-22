# Paying for Your Own Data: A Closed-Loop Benchmark for Scientific ML Agents

*Draft. Sections 1–5 and 9 are written; Sections 6–8 carry `«FILL: …»`
tokens wherever a number depends on the running campaign
(`experiments/matrix-v3-n10-20260917`). Nothing is asserted before it is
measured — grep for `«FILL` to find every open slot.*

---

## Abstract

Benchmarks for coding and ML agents overwhelmingly hand the agent its data.
We present a benchmark in which the agent must **generate its own data from
an expensive physics simulator under a fixed solve budget**, then ship a
predictor that is scored independently on a frozen grid it never sees. The
task is surrogate modelling of the Grad–Shafranov plasma equilibrium: an
agent must learn to drive a finite-element solver, design a sampling
campaign (an initial space-filling design plus up to three adaptive rounds
of exactly 100 solves), train a surrogate mapping five shaping parameters to
a 96×64 poloidal-flux field, and evaluate itself honestly. Every condition
emits the same machine-checkable contract, so any agent substrate is scored
identically.

We evaluate four agent harnesses at two access levels — library-assisted
(L2, may import a proven pipeline library) and from-scratch (L3, imports
audited and forbidden) — with **all cells on one model**, ten replicates
each, alongside a zero-LLM scripted baseline and an LLM-typed-decision
baseline that share the same pipeline code.

Our central finding concerns **verification**. «FILL: headline
passed-gates-but-physically-invalid rate and honesty-gap distribution from
§7». In a prior campaign one agent passed all nine contract gates while
scoring relative-L2 0.96 — roughly twice as bad as predicting the mean field
— because its predictor emitted zeros instead of NaN outside the plasma
boundary, and reported its own error as 0.0784 against an independently
scored 0.9596. Machine-checkable deliverable contracts, the standard
instrument in this literature, do not detect this class of failure. We
therefore separate three distinct questions a benchmark must ask — did the
agent deliver, is the artifact physically valid, and did it tell the truth
about its own results — and measure all three.

---

## 1. Introduction

### 1.1 The gap: agents that are given their data

Agentic ML benchmarks have converged on a shape: present the agent with a
dataset and a metric, and see how good a model it can fit. MLE-bench draws
on Kaggle competitions; MLAgentBench, DSBench, ScienceAgentBench and
related suites vary the domain and the scaffolding but keep the shape. The
data exists before the agent starts.

Real computational science does not look like that. The expensive step is
usually *acquiring* the data — running the simulation, the experiment, the
solve — and the interesting decision is *where to spend the next unit of
budget*. An agent that is good at fitting a given dataset may be useless at
deciding which equilibria are worth computing.

This benchmark restores that decision. The agent is given a solver, a
parameter space, and a budget, and must decide what to compute.

### 1.2 Why Grad–Shafranov

The ground-truth operator is TokaMaker (OpenFUSION Toolkit), a
finite-element solver for the axisymmetric Grad–Shafranov equation. It is a
good instrument for this purpose because it is:

- **genuinely expensive and genuinely failable** — extreme shaping drives it
  into isoflux fallback or outright non-convergence, so the feasible set is
  unknown to the agent and must be discovered;
- **high-dimensional in the output** (a 96×64 flux field, NaN outside the
  plasma boundary) but low-dimensional in the input (five parameters), so
  dimensionality reduction is a real design decision rather than a
  formality;
- **unforgiving about physics** — flux is signed and physical (Webers, not
  normalised), so an agent that quietly clips or renormalises produces a
  field that looks plausible and is wrong;
- **not memorisable** — the answer is not on the internet for these
  parameters.

### 1.3 Contributions

1. **A closed-loop scientific-ML benchmark** in which the agent pays for its
   own data from an expensive solver under a fixed budget, with a
   machine-checkable deliverable contract and independent scoring on a
   frozen grid (§3).
2. **A graded scaffolding ladder** (L0 scripted / L1 typed-decision / L2
   library-assisted / L3 from-scratch) with a matched zero-LLM baseline
   *inside the same contract*, which separates task difficulty from agent
   capability (§3.4).
3. **A model-controlled harness comparison**: four substrates on one
   identical model, so "which agent framework" is not confounded with
   "which model" — a confound we found in our own earlier campaigns and
   which is common in the literature (§5).
4. **A verification layer** distinguishing delivery, physical validity, and
   self-report honesty, with the finding that contract gates alone certify
   artifacts that are physically wrong (§7).
5. **Task characterisation** establishing that the task is hard for the
   right reasons, including an out-of-distribution collapse that makes the
   sampling decision consequential (§4).

---

## 2. Related work

«FILL: proper citations and positioning. The claim to defend is narrow and
should be stated narrowly — not "no benchmark has agents generate data"
(active learning and self-driving-lab work obviously does), but: *agentic
coding benchmarks* evaluate agents on datasets supplied in advance, and do
not make the acquisition policy part of what is scored. Check at minimum:
MLE-bench, MLAgentBench, DSBench, ScienceAgentBench, RE-Bench, CORE-Bench,
PaperBench, BixBench, DiscoveryBench, and the self-driving-lab /
LLM-for-experimental-design literature.»

---

## 3. The benchmark

### 3.1 The task

The agent receives a problem statement (frozen text, versioned) specifying:

- **The forward problem.** Fixed-boundary equilibria of D-shaped plasmas
  described by five parameters — major radius `r0 ∈ [0.35, 0.55]` m, minor
  radius `a ∈ [0.10, 0.20]` m, elongation `κ ∈ [1.0, 1.6]`, triangularity
  `δ ∈ [0.0, 0.4]`, plasma current `Ip ∈ [80, 200]` kA — with the solver's
  default profile shapes.
- **A campaign structure with a hard budget.** An initial space-filling
  design of at most 150 solves, then up to three adaptive rounds of exactly
  100 new solves each (failed solves count against the round). A one-shot
  design is explicitly insufficient: each round's points must be chosen
  adaptively from the current model or data.
- **A stopping criterion.** Keep a validation set separate from the test
  set; stop early once validation error falls to ≤ 0.30 × the
  mean-predictor baseline.
- **An honest evaluation requirement.** A held-out test set of at least 20
  solves at random parameters with a recorded seed, never used for
  training, model selection, or acquisition.
- **A self-assessment requirement.** Report whether the adaptive sampling
  actually helped, "or an honest statement that it did not".

Process gates are mandatory and identical for every condition: a meshing
milestone (reproduce the solver's own worked example before generalising), a
pilot gate (≥ 20 solves at ≥ 50% success before starting the rounds), a
storage-validation gate (a solve counts as successful only after its stored
artifact is re-read from disk and confirmed to contain finite values), and a
deliverable self-test (run every documented command in a fresh process
before declaring done).

### 3.2 The deliverable contract

Every condition emits the same artifacts, checked mechanically
(`src/autotokamak/bench/contract.py`):

| Gate | What it checks |
|---|---|
| `artifact:{report.json,predict.py,README.md}` | present and non-empty |
| `report_parses` | `report.json` is valid JSON |
| `report_keys` | `n_solves_{attempted,succeeded}` and `metrics.{test,baseline}_rel_l2.{mean,median,p90}` present |
| `no_autotokamak_import` | **L3 only** — no workspace `.py` imports the platform library, by regex over plain imports, `importlib`, and `__import__` |
| `predict_runs` | `python predict.py --input params.json --output pred.npz` exits 0 |
| `predict_shape` | returned `psi` is `(N, 96, 64)` |
| `predict_grid` | returned `R`, `Z` match the frozen axes to 1e-9 |

`contract.passed` is the flat AND of all gates: no weighting, no partial
credit. The scoring subprocess is stripped of provider credentials, so
`predict.py` must work offline, and runs in its own process group so a
timeout reaps the solver children too.

**`report_keys` deliberately checks key *presence*, never correctness.** This
is what makes §7's honesty measurement possible: the agent's self-reported
metrics are recorded verbatim and never cross-checked by the contract, so
they can be compared against independent scoring afterwards.

### 3.3 Independent scoring

Predictions are scored against a frozen test set of 60 parameter vectors
(committed, seed 20260809) solved once and stamped with provenance. The
metric is per-sample relative L2 over the finite ground-truth points:

    err_i = ||psi_pred_i − psi_true_i||₂ / ||psi_true_i||₂

with the masked mean taken only where ground truth is finite. **A NaN
prediction at a finite ground-truth point is scored as a full-magnitude
miss** (prediction treated as zero), never skipped — otherwise an all-NaN
prediction scores as perfect. Accuracy is quoted against the trivial
mean-flux-map predictor on the same set: `100 × (1 − rel_l2 / baseline)`,
where the baseline scores 0.4933.

**The inverse blind spot, and the full-grid metric.** Masking to finite
ground truth also means the metric ignores field predicted *outside* the
plasma, where there is none. That is not a corner case: one run reported
0.1142 while emitting a near-full-magnitude spurious plasma in the vacuum
region, whose honest error is **1.94** — four times worse than the
baseline. We therefore also report a **full-grid** relative L2 that charges
for exterior field at the same denominator, and their ratio
(`exterior_inflation`) as the violation's magnitude. The contract metric is
unchanged, so archived scores stay comparable; the full-grid figure is
recorded beside it.

### 3.4 The access-level ladder

| Level | Agent's role | Library |
|---|---|---|
| **L0** scripted | none — seeded heuristic decisions, zero LLM calls | full |
| **L1** structured | LLM makes only typed decisions; pipeline code fixed | full |
| **L2** library-assisted | agent writes the glue code | may import |
| **L3** from-scratch | agent writes everything | **forbidden, audited** |

L0 is the golden baseline: same seed, same decisions, no model. Having a
zero-LLM condition *inside the same contract and scored on the same frozen
set* is what lets us say how much of a score is the task and how much is the
agent.

### 3.5 Harnesses

Four agent substrates, one adapter each, all implementing the same
`run(task, workspace, run_dir, model, timeout) → RunResult` interface:
`ursa` (a plan/execute agent pair on LangGraph), `dspy` (plan → per-step
ReAct → review → fix), `pi` and `cursor` (CLI coding agents). A no-LLM
`echo` mock exercises the machinery in CI.

---

## 4. Task characterisation: is this hard for the right reasons?

Before asking whether agents can do the task, we establish what a competent
non-agentic solution achieves and where the difficulty lies. These results
come from a direct sweep (pool n = 9993, test n = 1499), independent of any
agent. Error is reported as a ratio to the mean-predictor baseline, so 1.0
is "no better than predicting the average field".

**Scale.** Error falls steadily with training-set size and saturates:

| N | 50 | 100 | 200 | 400 | 800 | 1600 | 3200 | 6400 |
|---|---|---|---|---|---|---|---|---|
| best ratio | 0.575 | 0.526 | 0.485 | 0.465 | 0.369 | 0.303 | 0.278 | 0.268 |
| R² | 0.804 | 0.840 | 0.865 | 0.876 | 0.922 | 0.948 | 0.956 | 0.959 |

The best model family **changes with N** — polynomial ridge wins up to
N = 200, kernel ridge from N = 400 on. Model selection is therefore a real
decision that depends on the data the agent chose to collect, not a fixed
answer. Note also that the agents' budget (≤ 450 campaign solves) sits in
the steep part of this curve, which is deliberate: it is a regime where
sampling choices still matter.

**Where you sample matters more than how much, at small N.** At a fixed
budget, comparing space-filling, uniform-random, and deliberately clustered
designs:

| N | space-filling | random | clustered |
|---|---|---|---|
| 100 | 0.550 | 0.699 | 0.963 |
| 400 | 0.358 | 0.438 | 0.873 |
| 1600 | 0.289 | 0.309 | 0.808 |

Space-filling beats random by 21% at N = 100, narrowing to 6% by N = 1600;
clustered sampling is catastrophic at every budget. Coverage is the
first-order sampling concern, and its value decays with budget.

**Dimensionality reduction.** Reconstruction error bottoms out around
k ≈ 24 PCA components (ratio 0.293) and degrades slightly by k = 32,
confirming the field is low-rank but not as low-rank as normalised flux
would be — retaining physical Webers costs rank and is the right choice
anyway, because per-sample normalisation destroys the `Ip` dependence
entirely.

**Out-of-distribution collapse.** Splitting the parameter space and training
on one half:

| train → test | ratio | R² |
|---|---|---|
| low → low | 0.376 | 0.927 |
| high → high | 0.401 | 0.919 |
| **low → high** | **0.943** | **0.280** |
| **high → low** | **0.937** | **−0.097** |

Outside its training region the surrogate is no better than the mean
predictor, and in one direction worse than useless (negative R²). This is
the result that makes the benchmark's central decision consequential: an
agent that samples badly does not merely lose a few percent, it produces a
model that fails completely off-distribution — and, because it evaluates
itself on its *own* test set, it may never notice.

---

## 5. Experimental design

**Conditions.** `{L2, L3} × {ursa, dspy, pi, cursor}` = 8 cells, **ten
replicates each** (80 runs), plus L0 and L1 reference rows from the
pre-written pipeline.

**Model control.** Every cell runs the *same* model (gpt-5.2). This matters:
in our own earlier campaign one harness ran a different model family from
the others, making "harness effect" and "model effect" inseparable. Cursor
takes bare model ids while the other adapters take prefixed ones, so the
model is pinned per-harness in the task specification rather than passed as
a single override.

**Equal compute budgets, and why they are hard to enforce.** Fair
comparison requires every substrate to get the same wall-clock budget. This
turned out to be the single hardest engineering requirement in the
benchmark, and we report the failure modes because they generalise to
anyone building an agent evaluation.

Two adapters originally accepted a `timeout_seconds` argument and silently
ignored it, so they ran 33–98 minutes against 8–13 for the others. Fixing
that revealed two deeper problems, both discovered only by running the full
campaign:

0. **Outer feedback rounds were a two-tier system.** `feedback_rounds: 2`
   is honoured only by the two adapters that implement an outer review-and-
   fix loop; the CLI-session substrates ignore the field entirely. Two
   harnesses were therefore getting a second corrective pass the others
   never got — the mechanical cause of their longer runtimes and ~6x cost.
   It is now **1 for every harness**, the only value all adapters honour
   identically.

1. **A timeout raised as an ordinary exception is swallowed.** Agent
   frameworks wrap their step loops in broad `except Exception` handlers and
   treat anything caught as a recoverable step failure. A timeout delivered
   by signal therefore vanishes into the agent's own retry logic, and
   because `signal.alarm` is one-shot, a single swallowed alarm disables the
   cap permanently. One harness absorbed its 90-minute cap this way and ran
   for **over ten hours**. The fix is to raise a `BaseException` subclass,
   for the same reason `KeyboardInterrupt` is one.
2. **Killing the agent process does not stop the work.** Python's
   `subprocess.run(capture_output=True, timeout=...)` kills the direct child
   and then waits on its pipes again — but solver grandchildren inherit
   those pipes and hold them open, so the "timeout" path itself blocks
   indefinitely. One cell sat in this deadlock for **9 hours 19 minutes**.
   The fix is to run each agent in its own process group and signal the
   group.

The general lesson: **an agent harness is third-party code and cannot be
trusted to honour a deadline from inside.** Our final design enforces the
budget at three independent levels — in-adapter (graceful, and still
records cost and partial output), process-group kill, and a driver-side
watchdog that cannot be intercepted by anything running within the cell.
Benchmarks that enforce time limits only in-process will silently grant
unequal compute to whichever substrate is least well-behaved, which is
precisely backwards.

**What is held fixed.** Prompt text (version 3), the frozen evaluation grid,
the 60-parameter test set, the contract gate set, and the scoring rule.
Results are pooled only within one prompt version and one scoring epoch;
the aggregation tool refuses to mix versions without an explicit override.

**Analysis.** The unit of analysis is the cell, not the run.

- Contract pass rate per cell with **Wilson score intervals** (which, unlike
  the normal approximation, do not collapse to zero width at 0/n or n/n).
- Relative-L2 **medians with percentile bootstrap CIs** — not means: a
  single catastrophic cell (we have observed 0.96) makes a mean
  uninterpretable.
- **Primary contrast L2 vs L3, paired within harness**, since harness is a
  blocking factor rather than a treatment; significance by exact two-sided
  sign test, reported as descriptive given only four pairs.
- Per-cell orderings are reported as exploratory; the pooled level contrast
  is the claim that carries weight.

**Cost.** Recorded per run, measured from each substrate where it reports
dollars and otherwise derived from token counts and a committed, dated
price table. The cursor cell's figure is an estimate at provider rates from
token counts, not that vendor's actual billing, and is labelled derived.

---

## 6. Results

> Campaign `experiments/matrix-v3-n10-20260917` in progress. Populate from
> `python tools/aggregate_matrix.py --tag matrix-v3-n10-20260917`.

### 6.1 Main table

«FILL: per-cell table — n, contract pass k/10 with Wilson CI, rel-L2 median
with bootstrap CI, min/max, physically-valid k/n, cost.»

### 6.2 Does library access help? (L2 vs L3)

«FILL: pooled pass rates per level with Wilson CIs; paired-by-harness deltas
in rel-L2 median and pass rate; sign-test p. State plainly if the answer is
"no detectable difference" — with four harness pairs that is a likely and
perfectly reportable outcome.»

### 6.3 Does the harness matter?

«FILL: per-harness spread with the model held constant. This is the
model-controlled comparison; state the effect size, not just an ordering.»

### 6.4 Against the non-agentic baselines

«FILL: L0 (scripted, zero LLM) and L1 (typed decisions) rows on the same
frozen test set, versus the agent cells. The interesting quantity is whether
any agent condition beats the zero-LLM baseline, and at what cost.»

### 6.5 Cost and budget compliance

«FILL: cost per cell; cost per unit accuracy; distribution of
`n_solves_attempted` against the 450-solve campaign bound (validation and
test solves are additional and uncapped, so report the raw distribution
rather than a pass/fail).»

---

## 7. Verification: delivery, validity, and honesty

The benchmark separates three questions that the literature usually
collapses into one.

### 7.1 Passing the gates is not being right

`contract.passed` certifies delivery. It does not certify that the
artifact is physically meaningful. We record three structural diagnostics
beside the contract, never as gates (adding a gate would invalidate every
archived run):

- **Exterior inflation** — full-grid error ÷ interior error. Above ~1.5 the
  prediction is asserting plasma in vacuum.
- **Prediction spread ratio** — does the prediction vary across test inputs
  as much as ground truth does? A constant predictor that ignores its input
  is finite and correctly shaped too.
- **Sanity** — is relative-L2 below 0.8 on *both* metrics, i.e. is it a
  surrogate at all?

We judge validity on **magnitude, not pixel counts**, and the distinction is
load-bearing. NaN-mask agreement is recorded but not decisive, because two
runs in our own shakedown scored *identical* mask agreement of 0.195 and
were not remotely the same defect: one had exterior inflation 1.22 (small
values written outside the boundary — an interface-convention slip on a
model whose field is otherwise good) and the other 16.97 (a fabricated
exterior plasma). A pixel-count criterion marks both invalid; only the
second is.

A run that passes every gate and fails these is recorded as
`passed_gates_but_invalid`.

**The motivating case**, from a prior campaign: an agent passed **9/9 gates**
with relative-L2 **0.9596** — accuracy **−94.5%**, roughly twice as bad as
predicting the mean field. Its NaN-mask agreement was **0.195** (the
prediction contained no NaN at all, against ground truth that is 80.5% NaN)
and its spread ratio **0.027** (very nearly a constant predictor). Both
diagnostics identify the failure immediately; no contract gate does.

**The subtler case**, which is arguably the stronger result: an agent passed
9/9 gates reporting **0.1142**, a perfectly respectable score, while its
full-grid error was **1.9371**. The interior field was fine; it had
fabricated a plasma in the vacuum region, and the contract metric — by
construction — could not see it. The failure is invisible to the gates, to
the headline metric, and to the agent's own evaluation, and is revealed only
by scoring the region the metric excludes.

«FILL: rate of passed-gates-but-invalid across the 80-run campaign, by
level and by harness.»

### 7.2 The honesty gap

Because the contract checks that self-reported metrics are *present* but
never that they are *true*, the difference between an agent's own reported
error and its independently scored error is directly measurable. We define

    honesty_gap = self_reported_rel_l2_mean − independently_scored_rel_l2_mean

Negative values mean the agent claimed to be better than it was. The
motivating run above reported **0.0784** against an actual **0.9596** — a gap
of **−0.881**, claiming roughly twelvefold better accuracy than it achieved.
By contrast our pre-flight validation run reported 0.1398 against an actual
0.1316, a gap of **+0.008**: slightly *understating* its own performance.

«FILL: distribution of honesty_gap across the campaign — median, spread, and
the count of runs with |gap| beyond some stated threshold. Report by level
and harness. Note whether large gaps co-occur with invalid artifacts or
appear independently, which are different pathologies.»

### 7.3 Failure taxonomy

«FILL: categorise every non-passing run. Seed categories observed
previously: solver never driven successfully; predictor interface broken;
physics corrupted (e.g. filtering out negative flux); adaptive sampling
claimed but absent; run killed by a single transient API error. Include the
blind LLM-judge's structured decision-style and adaptive-sampling
classifications, and cross-check them against the static code metrics.»

### 7.4 Did adaptive sampling help?

«FILL: the forced-action control arm (`tools/run_control_arm.py`) — forced
adaptive acquisition versus a blind space-filling append, matched solve
budget, scored on worst-cell accuracy over an envelope eval set. The prior
is not favourable: the single previous run that exercised adaptive
enrichment made the model worse. A clean negative is a result.»

---

## 8. Limitations

- **L3 receives two scaffolding instructions L2 does not** (a mandatory
  meshing milestone and a warning against hand-built meshes). These are
  arguably necessary — L3 agents must write solver plumbing that L2 agents
  can delegate to a library call — but it means the L2/L3 contrast bundles
  library access with a prompt difference, and should not be read as a pure
  library-access effect.
- **Four harness pairs** give the paired contrast very little power. The
  pooled level comparison is the defensible claim; per-cell orderings are
  exploratory.
- **One model.** Model control removes a confound but buys it with external
  validity: we cannot say these results hold for other models.
- **Wall-clock time is not a clean metric here** — the campaign ran several
  cells concurrently on one machine and the solver is CPU-bound, so
  reported wall time reflects contention. Cost and turn counts are the
  efficiency axes we report.
- **The solve budget is self-policed, and routinely exceeded.** Nothing
  enforces the ≤450-solve campaign bound; agents report their own counts. In
  the shakedown, six of eight runs exceeded it and usage spanned 205–604, a
  3x range. Raw accuracy therefore partly rewards whoever spent most
  physics, and we report **accuracy per 100 solves** alongside it. (The
  direction is not the obvious one: the best cell used the fewest solves.)
- **Timeout is an operational cap, not a task constraint.** The problem text
  states no time budget, so agents cannot triage against one, and a timed-out
  run is not a failed one — one was killed holding 9/9 gates, a valid
  artifact and the second-best score. Timeouts are reported as a separate
  outcome, never folded into the pass rate.
- **Tool access is not identical across substrates.** One adapter explicitly
  blocks web tools, one runs with no allowlist at all, and the others expose
  differing named tool sets. We audited the event streams and found no web
  usage in practice, but the affordances differ by construction.
- **Sampling parameters are not identical**: one substrate pins
  `temperature=1.0` while the others take provider defaults.
- **Unequal replicate counts.** One substrate routinely fails to terminate
  within the task's own 90-minute budget, so replicating it ten times would
  spend a large share of the campaign re-recording the same timeout. It is
  run at lower n and reported as a documented non-completer; pass-rate
  intervals account for the smaller denominator, and the paired contrast
  uses each harness's own cells.
- **The `feedback_rounds` setting binds only two of the four adapters**; the
  others run a single session.
- **README-vs-report consistency checking is heuristic** and advisory only;
  the qualitative call is left to the blind LLM judge.
- **`claude_sdk` is excluded** from this campaign: the model-control
  decision was to standardise on one provider's models, and that adapter
  cannot run them.
- **Judge self-preference.** «FILL: state the judge model and whether it
  shares a family with any evaluated condition; report the cross-judge
  agreement check.»
- The benchmark is **fixed-boundary, single profile family, five
  parameters**. It is a real physics problem but not a full one.

---

## 9. Conclusion

«FILL: write last, from §6 and §7.»

The durable point is independent of how the numbers land. A benchmark that
checks only whether an agent produced the required files will certify work
that is physically wrong and self-reported dishonestly, and it will do so
silently. Separating *did it deliver*, *is it valid*, and *did it tell the
truth* costs little — three diagnostics computed from predictions the
benchmark already has — and it is the difference between measuring agent
capability and measuring agent compliance.

---

## Reproducibility

```bash
pip install -e ".[ml,dev,harnesses]"
python -m autotokamak.bench freeze-testset          # ground truth (committed params)
tools/run_campaign.sh --tag <tag> --reps 10 --parallel 3 \
                      --harnesses "ursa dspy pi cursor"
python tools/aggregate_matrix.py --tag <tag>        # main table
python tools/cost_report.py       --tag <tag>
python tools/eval_code_metrics.py --tag <tag>
python tools/judge_code.py compare --tag <tag> --samples 3
python tools/run_control_arm.py   --reps 5          # §7.4
```

Task texts are frozen and versioned; every run records the task's SHA-256,
its prompt version, its model, and its replicate index. The frozen test set
carries provenance attributes (solver version, parameter and grid hashes,
scoring epoch) and is never regenerated.
