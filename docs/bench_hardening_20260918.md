# Bench hardening, 2026-09-18 — timeout enforcement and the verification layer

Written after the first attempt at the replicated prompt-v3 campaign
(`experiments/matrix-v3-n10-20260917`) stalled: **4 of 80 runs completed in
10 hours**. Companion to `docs/bench_hardening_20260818.md`.

## What went wrong

Three compounding defects, none of which appear in a short smoke test —
all five harnesses had passed `smoke.yaml` and a full L3-pi cell an hour
earlier.

### 1. A timeout raised as `Exception` is swallowed by the agent

`harnesses/base.py:time_limit` delivered the cap via `SIGALRM`, raising
`HarnessTimeout(Exception)`. URSA (like most agent frameworks) wraps its
step loop in a broad `except Exception` and treats anything caught as a
recoverable step failure, so the timeout was absorbed into its retry logic.
`signal.alarm` is **one-shot**, so that single swallow disabled the cap
permanently. `L3-ursa-r1` ran **10h18m** against a 90-minute budget, and the
string `HarnessTimeout` appears nowhere in its 5.4 MB log.

**Fix:** `HarnessTimeout` now subclasses `BaseException`, for exactly the
reason `KeyboardInterrupt` does; the handler also re-arms the alarm at 30s
so a future swallow degrades the cap to "late" rather than "gone".

### 2. `subprocess.run(capture_output=True, timeout=...)` deadlocks

On timeout, `subprocess.run` kills the direct child and then calls
`communicate()` again. That second call blocks until every writer to the
pipes closes — and the agent's solver grandchildren inherit stdout/stderr
and keep them open. The timeout path itself hangs forever. `L3-pi-r2` sat in
this state for **9h19m**; `cursor.py` carried the identical pattern, and
matters more there because that adapter already documents a known
`cursor-agent -p` hang mode.

**Fix:** both adapters now use `Popen(start_new_session=True)` +
`communicate(timeout)` + `os.killpg(SIGKILL)`, the pattern
`bench/contract.py:run_predict` already used correctly. Partial output and
token usage are still harvested from a killed run, so a timed-out cell
still reports the budget it burned.

### 3. No enforcement outside the agent's own process

Both defects above live *inside* code we do not control. An agent harness is
third-party software and cannot be trusted to honour a deadline from within.

**Fix:** `tools/run_campaign.sh` now runs each cell in its own process group
(bash job control; macOS ships no `setsid(1)`) with a watchdog that
`kill`s the group after `CELL_HARD_TIMEOUT` (default 6600s, deliberately
above the task's own 5400s so the graceful in-adapter path gets first
refusal). Verified against a SIGTERM-ignoring parent holding a long-lived
grandchild: killed in 6s, no survivors.

**Net effect:** the budget is now enforced at three independent levels —
in-adapter, process-group, driver watchdog.

## Knock-on consequences

- **ursa runs at reduced n.** It did complete in 55 min under v2, but under
  v3 it does not reliably terminate. Replicating it ten times would spend a
  large share of the campaign re-recording the same timeout, so
  `run_campaign.sh` accepts a per-harness rep cap (`"ursa:4"`), and ursa is
  reported as a documented non-completer.
- **Cost recovery.** Killed runs write no `result.json`, so their spend is
  invisible to `tools/cost_report.py`. URSA writes per-call accounting into
  `workspace/ursa_metrics/*.json` (`costs.total_usd`), from which the two
  killed runs were recovered at $12.31. Total spend on the aborted attempt:
  **~$21** (plus ~$3 pre-flight).
- **Derived costs reach the aggregate.** `tools/aggregate_matrix.py` now
  falls back to `cost_report.csv` when a substrate does not self-report
  dollars, so cursor cells are no longer blank on the efficiency axis.

## What the four salvaged runs already show

All four passed every contract gate. Two were not physically valid.

| run | gates | rel-L2 | mask agree | valid | honesty gap | solves |
|---|---|---|---|---|---|---|
| L3-dspy r1 | 9/9 | 0.0585 | 0.9994 | yes | +0.008 | 320 |
| L3-dspy r2 | 9/9 | 0.1942 | **0.635** | **no** | +0.006 | 305 |
| L3-cursor r1 | 9/9 | 0.1659 | 0.9996 | yes | −0.011 | **520** |
| L3-pi r1 | 9/9 | **5.2647** | 0.9996 | **no** | **−5.22** | 290 |

- **Replicate variance is real and large**: the same cell (L3-dspy) scored
  0.0585 and 0.1942 on consecutive runs, a 3.3x spread. No n=1 ordering in
  any earlier campaign is defensible.
- **Two distinct invalidity pathologies**, which a single scalar error
  cannot separate: dspy r2 is a *masking* failure (NaN-mask agreement 0.635)
  while pi r1 has a near-perfect mask but a *scale* error putting it 10x
  worse than the mean predictor.
- **Budget overrun is measurable under v3**: cursor used 520 solves against
  a 450-solve campaign bound (validation/test solves are additional and
  uncapped, so this is indicative, not a violation).

## Still open

- `dspy_harness.py` runs jailed shell commands with `MAX_SHELL_TIMEOUT`
  (4h) internally; the outer alarm and the driver watchdog bound it, but the
  inner value is still larger than any task budget.
- README-vs-report consistency checking (`bench/diagnostics.py`) is
  heuristic and advisory; the blind LLM judge remains the qualitative
  instrument.
- Wall-clock time remains contaminated by run-level parallelism and is not
  a reportable efficiency metric; cost and turn counts are.
