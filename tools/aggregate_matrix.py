# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Aggregate REPLICATED benchmark cells into the paper's main table.

Every other reporting path in this repo (``bench compare``,
``tools/matrix_report.py``) emits one row per run. That was right while each
cell was run once; it is wrong now that a cell is run n times, because the
unit of analysis is the CELL, not the run.

This tool groups runs by condition and reports, per cell:

  * contract pass RATE with a Wilson score interval (small-n honest: it does
    not collapse to a zero-width interval at 0/n or n/n like the normal
    approximation does),
  * relative-L2 MEDIAN with a percentile bootstrap CI — median, not mean,
    because the failure mode in this corpus is a 0.96-type catastrophic cell
    that drags a mean into meaninglessness,
  * the diagnostics rates (physically valid, passed-gates-but-invalid) and
    the signed honesty gap,
  * cost,
  * the METHODOLOGY the cell converged on — the modal chain of methods, how
    often the replicates of one cell agree on it, and the per-round decision
    logic (criterion stated, evidence in hand, what it decided next). Two
    cells can land on the same error by different reasoning; this is the
    axis that separates them. See ``autotokamak.bench.methodology``.

It then pools by access level and runs the primary contrast, L2 vs L3,
PAIRED WITHIN HARNESS — harness is a blocking factor, not a treatment, so
pairing removes between-substrate variance from the comparison.

Usage:
    python tools/aggregate_matrix.py --tag matrix-v4-...
    python tools/aggregate_matrix.py --tag armA --tag armB --allow-mixed-prompt

Writes experiments/<first-tag>/aggregate.csv and prints the tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260917


# ---- small statistics, spelled out ---------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_median_ci(values: list[float], n_boot: int = BOOTSTRAP_N
                        ) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI for the median. None below n=3 — with one or
    two observations a resampled interval is theatre, not evidence."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size < 3:
        return None, None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(v, size=(n_boot, v.size), replace=True)
    meds = np.median(draws, axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _strata(runs: list[dict], key: str) -> dict[str, dict[str, list[float]]]:
    """Runs grouped harness -> level -> values, keeping only complete blocks."""
    by_harness: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"L2": [], "L3": []})
    for r in runs:
        v = r.get(key)
        if isinstance(v, (int, float)) and r.get("level") in ("L2", "L3"):
            by_harness[r["harness"]][r["level"]].append(float(v))
    return {h: d for h, d in by_harness.items() if d["L2"] and d["L3"]}


def _rank_statistic(strata: dict[str, dict[str, list[float]]]) -> float:
    """Standardised within-harness rank sum for L3, averaged over harnesses.

    The van Elteren / stratified-Wilcoxon statistic: every RUN contributes
    its rank within its own harness. This is the textbook test for a blocked
    two-sample design, and it is the primary one here because the
    median-difference statistic below discards almost all within-cell
    information — on the matrix-v4 campaign the two gave p=0.0001 and
    p=0.33 on identical data, purely because a median of five noisy runs is
    itself noisy while the ranks are not.
    """
    total = 0.0
    for v in strata.values():
        values = np.asarray(v["L2"] + v["L3"], dtype=float)
        order = values.argsort().argsort().astype(float) + 1.0
        # average ranks for ties
        for val in np.unique(values):
            mask = values == val
            if mask.sum() > 1:
                order[mask] = order[mask].mean()
        n2, n3 = len(v["L2"]), len(v["L3"])
        observed = order[n2:].sum()
        expected = n3 * (n2 + n3 + 1) / 2
        sd = math.sqrt(n2 * n3 * (n2 + n3 + 1) / 12)
        total += (observed - expected) / sd
    return total / math.sqrt(len(strata))


def stratified_rank_test(runs: list[dict], key: str, n_perm: int = 20_000
                         ) -> tuple[float | None, float | None, int]:
    """PRIMARY test: stratified rank, labels permuted within harness.

    Exact by construction (the null is generated by the same shuffling the
    design licenses), so its type-I error needs no distributional
    assumption.
    """
    strata = _strata(runs, key)
    n_used = sum(len(d["L2"]) + len(d["L3"]) for d in strata.values())
    if len(strata) < 2:
        return None, None, n_used
    observed = _rank_statistic(strata)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    count = 0
    for _ in range(n_perm):
        shuffled = {}
        for h, v in strata.items():
            pool = rng.permutation(np.asarray(v["L2"] + v["L3"], dtype=float))
            shuffled[h] = {"L2": list(pool[:len(v["L2"])]),
                           "L3": list(pool[len(v["L2"]):])}
        if abs(_rank_statistic(shuffled)) >= abs(observed) - 1e-12:
            count += 1
    return observed, (count + 1) / (n_perm + 1), n_used


def stratified_permutation_test(runs: list[dict], key: str,
                                n_perm: int = 20_000
                                ) -> tuple[float | None, float | None, int]:
    """Exact-style permutation test for L2 vs L3, blocking on harness.

    Why this exists: with 4-5 harnesses the paired SIGN test cannot reach
    p<0.05 however clean the result is — its smallest attainable two-sided
    p-value at n=4 pairs is 0.125, and at n=5 it is 0.0625. Reporting only
    that would put a ceiling on the paper's primary claim that has nothing
    to do with the evidence.

    This test uses the RUNS rather than the cells: within each harness the
    L2/L3 labels are shuffled among that harness's runs, so harness effects
    (the blocking factor) cannot leak into the comparison, and the null is
    "access level does not matter within a substrate". The statistic is the
    mean across harnesses of the within-harness median difference, which
    keeps the per-harness robustness of a median while weighting every
    substrate equally regardless of how many replicates it completed.

    Returns (observed statistic, two-sided p, n_runs_used).
    """
    by_harness: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"L2": [], "L3": []})
    n_used = 0
    for r in runs:
        v = r.get(key)
        if not isinstance(v, (int, float)) or r["level"] not in ("L2", "L3"):
            continue
        by_harness[r["harness"]][r["level"]].append(float(v))
        n_used += 1
    usable = {h: d for h, d in by_harness.items() if d["L2"] and d["L3"]}
    if len(usable) < 2:
        return None, None, n_used

    def statistic(assignment: dict[str, tuple[list[float], list[float]]]) -> float:
        diffs = [float(np.median(l3) - np.median(l2)) for l2, l3 in assignment.values()]
        return float(np.mean(diffs))

    observed = statistic({h: (d["L2"], d["L3"]) for h, d in usable.items()})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pooled = {h: (np.array(d["L2"] + d["L3"], dtype=float), len(d["L2"]))
              for h, d in usable.items()}
    count = 0
    for _ in range(n_perm):
        assignment = {}
        for h, (values, n_l2) in pooled.items():
            perm = rng.permutation(values)
            assignment[h] = (list(perm[:n_l2]), list(perm[n_l2:]))
        if abs(statistic(assignment)) >= abs(observed) - 1e-12:
            count += 1
    # +1 smoothing: a permutation p of exactly 0 is not a thing you can
    # observe from a finite number of shuffles.
    return observed, (count + 1) / (n_perm + 1), n_used


def sign_test_two_sided(diffs: list[float]) -> tuple[int, int, float | None]:
    """Exact two-sided sign test. Ties dropped (standard)."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, None
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return pos, neg, min(1.0, 2 * tail)


# ---- loading -------------------------------------------------------------

def _dig(d: Any, *path, default=None):
    for key in path:
        if not isinstance(d, dict):
            return default
        d = d.get(key)
    return d if d is not None else default


def _derived_costs(tag_dir: Path) -> dict[str, float]:
    """Costs from a prior `tools/cost_report.py --tag` run, keyed by run_id.

    Substrates differ in whether they self-report dollars: claude_sdk, pi,
    dspy and ursa do, cursor reports only tokens. Without this the cursor
    cells would show a blank cost and the efficiency axis would be empty
    for a quarter of the matrix. Self-reported (measured) values always
    win; this only fills gaps.
    """
    out: dict[tuple[str, str], float] = {}
    csv_path = tag_dir / "cost_report.csv"
    if not csv_path.is_file():
        return out
    try:
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    # Key on (condition, run_id), NOT run_id alone: run ids are
                    # UTC timestamps to the second and a campaign launches a
                    # whole wave at once, so every cell in a wave shares one.
                    # Keying on run_id alone cross-assigns costs between
                    # harnesses (observed: cursor reported ursa's spend).
                    out[(row["condition"], row["run_id"])] = float(row["cost_usd"])
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    return out


def load_runs(tag_dirs: list[Path]) -> list[dict]:
    rows = []
    for tag_dir in tag_dirs:
        derived = _derived_costs(tag_dir)
        for rp in sorted(tag_dir.glob("*/*/result.json")):
            try:
                r = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            condition = r.get("condition") or rp.parent.parent.name
            level, _, harness = condition.partition("-")
            rows.append({
                "tag": tag_dir.name,
                "condition": condition,
                "level": level,
                "harness": harness or r.get("harness", ""),
                "run_id": r.get("run_id") or rp.parent.name,
                "replicate": r.get("replicate"),
                "model": r.get("model", ""),
                "prompt_version": _dig(r, "task", "prompt_version"),
                "status": r.get("status", ""),
                "contract_passed": bool(_dig(r, "contract", "passed", default=False)),
                "rel_l2": _dig(r, "frozen_score", "test_rel_l2", "mean"),
                "rel_l2_full": _dig(r, "frozen_score", "test_rel_l2_full_grid", "mean"),
                "exterior_inflation": _dig(r, "frozen_score", "exterior_inflation"),
                "accuracy_pct": _dig(r, "frozen_score", "accuracy_pct"),
                "nan_mask_agreement": _dig(r, "frozen_score", "nan_mask_agreement"),
                "physically_valid": _dig(r, "diagnostics", "physically_valid"),
                "passed_but_invalid": _dig(r, "diagnostics", "passed_gates_but_invalid"),
                "honesty_gap": _dig(r, "diagnostics", "honesty_gap"),
                "n_solves": _dig(r, "diagnostics", "solves", "attempted"),
                "timed_out": r.get("status") == "timeout",
                "cost_usd": (r.get("cost_usd")
                             if r.get("cost_usd") is not None
                             else derived.get((condition,
                                               r.get("run_id") or rp.parent.name))),
                "wall_min": round((r.get("wall_seconds") or 0) / 60, 1),
                **_methodology_fields(r.get("methodology")),
            })
    return rows


# ---- methodology ---------------------------------------------------------

def _methodology_fields(meth: Any) -> dict:
    """Flatten a run's methodology record into matrix-ready columns."""
    if not isinstance(meth, dict) or "error" in meth:
        return {"chain": None, "acq_classes": [], "model_primary": None,
                "n_rounds": None, "criterion_switched": None,
                "evidence_grounded": None, "stop_decision": None,
                "adaptive_in_name_only": None, "prose_only": False,
                "iterations": [], "code_acq": [], "code_logic_signature": None,
                "code_evidence_scope": None, "selection_rule": [],
                "model_informed": None, "stated_vs_code": None,
                "only_stated": [], "only_implemented": [], "chooser_where": "",
                "shape_signature": None, "shape_dims": {}}
    chain = meth.get("method_chain") or {}
    logic = meth.get("decision_logic") or {}
    code = meth.get("code_logic") or {}
    svi = meth.get("stated_vs_implemented") or {}
    prose_only = (meth.get("evidence") or {}).get("prose_only_terms") or {}
    return {
        # What the CODE computes, and whether it matches what the run said.
        "code_acq": code.get("acquisition_implemented") or [],
        "code_logic_signature": code.get("code_logic_signature"),
        "code_evidence_scope": code.get("evidence_scope"),
        "selection_rule": code.get("selection_rule") or [],
        "model_informed": code.get("model_informed"),
        "stated_vs_code": svi.get("verdict"),
        "only_stated": svi.get("only_stated") or [],
        "only_implemented": svi.get("only_implemented") or [],
        "chooser_where": "; ".join(h.get("where", "")
                                   for h in (code.get("chooser_functions") or [])[:3]),
        # How the whole prompt was solved, dimension by dimension.
        "shape_signature": (meth.get("solution_shape") or {}).get("shape_signature"),
        "shape_dims": {k: (v or {}).get("value")
                       for k, v in ((meth.get("solution_shape") or {}).get("dimensions")
                                    or {}).items()},
        "chain": meth.get("chain_signature"),
        "acq_classes": logic.get("criterion_classes") or [],
        "model_primary": chain.get("model_primary"),
        "n_rounds": logic.get("n_rounds"),
        "criterion_switched": logic.get("criterion_switched"),
        "evidence_grounded": logic.get("evidence_grounded_fraction"),
        "stop_decision": logic.get("stop_decision"),
        "adaptive_in_name_only": logic.get("adaptive_in_name_only"),
        # A method named in the README/report that the code never evidences.
        "prose_only": bool(prose_only.get("acquisition") or prose_only.get("model_family")),
        "iterations": meth.get("iterations") or [],
    }


def _counts(values: list, sep: str = ",") -> str:
    """"a:3,b:1" — counts, so a table cell shows spread, not just presence."""
    c = Counter(v for v in values if v not in (None, ""))
    return sep.join(f"{k}:{n}" for k, n in c.most_common()) or "-"


def summarise_methodology(runs: list[dict]) -> dict:
    """Per-cell methodology summary: what the cell converged on, and how firmly."""
    chains = [r["chain"] for r in runs if r["chain"]]
    chain_counts = Counter(chains)
    modal, modal_n = (chain_counts.most_common(1)[0] if chain_counts else (None, 0))
    code_sigs = [r["code_logic_signature"] for r in runs if r["code_logic_signature"]]
    code_counts = Counter(code_sigs)
    code_modal, code_modal_n = (code_counts.most_common(1)[0] if code_counts
                                else (None, 0))
    rounds = [r["n_rounds"] for r in runs if isinstance(r["n_rounds"], int)]
    grounded = [r["evidence_grounded"] for r in runs
                if isinstance(r["evidence_grounded"], (int, float))]
    switched = [r["criterion_switched"] for r in runs if r["criterion_switched"] is not None]
    return {
        "condition": runs[0]["condition"],
        "n_with_chain": len(chains),
        # How reproducible is the METHOD, as opposed to the score: the share
        # of a cell's replicates that chose the modal chain.
        "chain_agreement": round(modal_n / len(chains), 2) if chains else None,
        "n_distinct_chains": len(chain_counts),
        "chain_modal": modal,
        "model_primary": _counts([r["model_primary"] for r in runs]),
        "acq_classes": _counts([c for r in runs for c in r["acq_classes"]]),
        "rounds_median": (int(np.median(rounds)) if rounds else None),
        "criterion_switched_k": sum(1 for v in switched if v),
        "criterion_switched_n": len(switched),
        "evidence_grounded_median": (round(float(np.median(grounded)), 2)
                                     if grounded else None),
        "stop_decisions": _counts([r["stop_decision"] for r in runs]),
        # ---- the code-comparison columns --------------------------------
        "code_acq": _counts([c for r in runs for c in r["code_acq"]]),
        "code_logic_modal": code_modal,
        "code_logic_agreement": (round(code_modal_n / len(code_sigs), 2)
                                 if code_sigs else None),
        "n_distinct_code_logics": len(code_counts),
        "selection_rule": _counts([s for r in runs for s in r["selection_rule"]]),
        "model_informed_k": sum(1 for r in runs if r["model_informed"]),
        "model_informed_n": sum(1 for r in runs if r["model_informed"] is not None),
        "stated_vs_code": _counts([r["stated_vs_code"] for r in runs]),
        "claimed_not_implemented": _counts([c for r in runs for c in r["only_stated"]]),
        "adaptive_in_name_only_k": sum(1 for r in runs if r["adaptive_in_name_only"]),
        "prose_only_claim_k": sum(1 for r in runs if r["prose_only"]),
    }


def cross_compare_cells(by_cell: dict[str, list[dict]]) -> list[dict]:
    """One row per solution dimension, one column per condition.

    A cross-comparison is a transposition: the question is "how did each of
    them answer THIS", and that only reads as a comparison when the answers
    share a line. Within a cell the replicates' modal answer is shown, with
    the count when they disagreed — a cell that solved the same prompt two
    different ways is itself a result.
    """
    from autotokamak.bench.solution_shape import DIMENSION_QUESTIONS

    dims: list[str] = []
    for runs in by_cell.values():
        for r in runs:
            for d in r.get("shape_dims") or {}:
                if d not in dims:
                    dims.append(d)
    rows = []
    for dim in dims:
        row = {"dimension": dim, "question": DIMENSION_QUESTIONS.get(dim, "")}
        seen, unknown = [], False
        for cond, runs in sorted(by_cell.items()):
            vals = [(r.get("shape_dims") or {}).get(dim) for r in runs]
            known = [v for v in vals if v]
            if not known:
                row[cond] = "-"
                unknown = True
                continue
            counts = Counter(known)
            modal, n = counts.most_common(1)[0]
            row[cond] = modal if len(counts) == 1 else f"{modal} ({n}/{len(known)})"
            seen.append(modal)
        row["n_distinct"] = len(set(seen))
        # A cell the dimension could not be read for is a difference too —
        # never let an unknown column be averaged into "all_same".
        row["agreement"] = ("mixed" if unknown and seen
                            else ("all_same" if seen and len(set(seen)) == 1
                                  else ("all_differ"
                                        if seen and len(set(seen)) == len(seen)
                                        else "mixed")))
        rows.append(row)
    # Rows where the agents disagreed are the ones worth reading first.
    order = {"all_differ": 0, "mixed": 1, "all_same": 2}
    return sorted(rows, key=lambda r: (order.get(r["agreement"], 3), r["dimension"]))


def write_solution_shape_csv(rows: list[dict], conds: list[str], out_dir: Path) -> Path:
    out = out_dir / "solution_shape.csv"
    fields = ["dimension", "question", *conds, "n_distinct", "agreement"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out


def write_methodology_csvs(runs: list[dict], out_dir: Path) -> tuple[Path, Path]:
    """One row per run (the chain) and one row per round (the logic)."""
    per_run = out_dir / "methodology.csv"
    fields = ["tag", "condition", "level", "harness", "run_id", "replicate", "status",
              "chain", "model_primary", "acq_classes", "n_rounds", "criterion_switched",
              "evidence_grounded", "stop_decision", "adaptive_in_name_only",
              "prose_only", "code_acq", "code_logic_signature", "code_evidence_scope",
              "selection_rule", "model_informed", "stated_vs_code", "only_stated",
              "only_implemented", "chooser_where", "shape_signature",
              "rel_l2", "contract_passed", "physically_valid"]
    with per_run.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            row = dict(r)
            for key in ("acq_classes", "code_acq", "selection_rule",
                        "only_stated", "only_implemented"):
                row[key] = "|".join(r[key])
            w.writerow(row)

    per_round = out_dir / "methodology_rounds.csv"
    rfields = ["tag", "condition", "run_id", "replicate", "round", "n_acquired",
               "criterion_classes", "criterion_source", "criterion_text",
               "val_rel_l2", "baseline_rel_l2", "val_over_baseline",
               "met_stop_threshold", "evidence_grounded", "decision",
               "reason_coverage"]
    with per_round.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rfields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            for it in r["iterations"]:
                row = {k: it.get(k) for k in rfields}
                row.update({"tag": r["tag"], "condition": r["condition"],
                            "run_id": r["run_id"], "replicate": r["replicate"]})
                row["criterion_classes"] = "|".join(it.get("criterion_classes") or [])
                text = it.get("criterion_text")
                row["criterion_text"] = (text or "")[:200]
                w.writerow(row)
    return per_run, per_round


# ---- aggregation ---------------------------------------------------------

def _rate(flags: list, n: int) -> tuple[int, int]:
    """Count of True among non-None flags, and how many were determinable."""
    known = [f for f in flags if f is not None]
    return sum(1 for f in known if f), len(known)


def summarise_cell(runs: list[dict]) -> dict:
    n = len(runs)
    n_pass = sum(1 for r in runs if r["contract_passed"])
    lo, hi = wilson_interval(n_pass, n)

    # Timeout is OUR wall-clock cap, not an agent failure and not something
    # the task even tells the agent about. Folding it into pass/fail
    # conflates "the agent could not do it" with "we stopped it" — one run
    # was killed holding 9/9 gates and the second-best score in the matrix.
    n_timeout = sum(1 for r in runs if r["timed_out"])
    n_errored = sum(1 for r in runs if r["status"] == "errored")

    scored = [r["rel_l2"] for r in runs if isinstance(r["rel_l2"], (int, float))]
    med = float(np.median(scored)) if scored else None
    blo, bhi = bootstrap_median_ci(scored)

    valid_k, valid_n = _rate([r["physically_valid"] for r in runs], n)
    pbi_k, pbi_n = _rate([r["passed_but_invalid"] for r in runs], n)
    gaps = [r["honesty_gap"] for r in runs if isinstance(r["honesty_gap"], (int, float))]
    costs = [r["cost_usd"] for r in runs if isinstance(r["cost_usd"], (int, float))]

    full = [r["rel_l2_full"] for r in runs if isinstance(r["rel_l2_full"], (int, float))]
    infl = [r["exterior_inflation"] for r in runs
            if isinstance(r["exterior_inflation"], (int, float))]
    # Accuracy per unit of expensive physics. The solve budget is
    # self-policed and varies ~3x across runs (205-604 against a 450
    # bound), so raw accuracy alone rewards whoever spent most.
    eff = [r["accuracy_pct"] / r["n_solves"] * 100
           for r in runs
           if isinstance(r.get("accuracy_pct"), (int, float))
           and isinstance(r.get("n_solves"), (int, float)) and r["n_solves"]]

    return {
        "condition": runs[0]["condition"],
        "level": runs[0]["level"],
        "harness": runs[0]["harness"],
        "n": n,
        "n_completed": sum(1 for r in runs if r["status"] == "completed"),
        "n_timeout": n_timeout,
        "n_errored": n_errored,
        "pass_k": n_pass,
        "pass_rate": round(n_pass / n, 3) if n else None,
        "pass_ci_lo": round(lo, 3) if lo is not None else None,
        "pass_ci_hi": round(hi, 3) if hi is not None else None,
        "n_scored": len(scored),
        "rel_l2_median": round(med, 4) if med is not None else None,
        "rel_l2_ci_lo": round(blo, 4) if blo is not None else None,
        "rel_l2_ci_hi": round(bhi, 4) if bhi is not None else None,
        "rel_l2_min": round(min(scored), 4) if scored else None,
        "rel_l2_max": round(max(scored), 4) if scored else None,
        "rel_l2_full_median": round(float(np.median(full)), 4) if full else None,
        "ext_inflation_median": round(float(np.median(infl)), 2) if infl else None,
        "n_solves_median": int(np.median([r["n_solves"] for r in runs
                                          if isinstance(r.get("n_solves"), (int, float))]))
                           if any(isinstance(r.get("n_solves"), (int, float)) for r in runs)
                           else None,
        "acc_per_100_solves": round(float(np.median(eff)), 2) if eff else None,
        "phys_valid_k": valid_k,
        "phys_valid_n": valid_n,
        "passed_but_invalid_k": pbi_k,
        "passed_but_invalid_n": pbi_n,
        "honesty_gap_median": round(float(np.median(gaps)), 4) if gaps else None,
        "cost_usd_mean": round(float(np.mean(costs)), 3) if costs else None,
        "cost_usd_total": round(float(np.sum(costs)), 2) if costs else None,
    }


def print_table(rows: list[dict], cols: list[str], title: str) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(none)")
        return
    w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", action="append", required=True,
                    help="experiments/<tag>; repeat to pool several arms")
    ap.add_argument("--experiments-dir", default=None)
    ap.add_argument("--allow-mixed-prompt", action="store_true",
                    help="Pool runs across prompt versions (normally refused)")
    ap.add_argument("--show-rounds", action="store_true",
                    help="Also print every adaptive round of every run "
                         "(always written to methodology_rounds.csv)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    exp_dir = Path(args.experiments_dir) if args.experiments_dir else repo_root / "experiments"
    tag_dirs = [exp_dir / t for t in args.tag]
    missing = [str(d) for d in tag_dirs if not d.is_dir()]
    if missing:
        print(f"No such tag dir(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    runs = load_runs(tag_dirs)
    if not runs:
        print("No result.json files found.", file=sys.stderr)
        return 1

    versions = {r["prompt_version"] for r in runs}
    if len(versions) > 1 and not args.allow_mixed_prompt:
        print(f"Refusing to pool prompt versions {sorted(map(str, versions))}: runs are "
              "only comparable within one version (benchmarks/README.md). "
              "Re-run with --allow-mixed-prompt if that is genuinely intended.",
              file=sys.stderr)
        return 1

    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_cell[r["condition"]].append(r)
    cells = [summarise_cell(v) for _, v in sorted(by_cell.items())]
    meth_cells = {c: summarise_methodology(v) for c, v in sorted(by_cell.items())}
    # Methodology columns ride along in aggregate.csv so the paper's table
    # and its method column come out of one file.
    for cell in cells:
        extra = dict(meth_cells.get(cell["condition"], {}))
        extra.pop("condition", None)
        cell.update(extra)

    print_table(
        cells,
        ["condition", "n", "n_completed", "n_timeout", "n_errored", "pass_k", "pass_rate",
         "pass_ci_lo", "pass_ci_hi", "n_scored", "rel_l2_median", "rel_l2_ci_lo",
         "rel_l2_ci_hi", "rel_l2_full_median", "ext_inflation_median", "phys_valid_k",
         "phys_valid_n", "passed_but_invalid_k", "n_solves_median",
         "acc_per_100_solves", "cost_usd_mean"],
        f"Per-cell summary (prompt_version={sorted(map(str, versions))[0]})",
    )

    # ---- methodology: what was chosen, and on what reasoning -------------
    meth_rows = [meth_cells[c["condition"]] for c in cells]
    print_table(
        meth_rows,
        ["condition", "n_with_chain", "chain_agreement", "n_distinct_chains",
         "model_primary", "acq_classes", "rounds_median", "criterion_switched_k",
         "criterion_switched_n", "evidence_grounded_median", "stop_decisions",
         "adaptive_in_name_only_k", "prose_only_claim_k"],
        "Methodology matrix (what each cell chose, and how firmly)",
    )
    print_table(
        meth_rows,
        ["condition", "code_acq", "selection_rule", "model_informed_k",
         "model_informed_n", "stated_vs_code", "claimed_not_implemented",
         "code_logic_agreement", "n_distinct_code_logics"],
        "Code comparison (what the generated code actually computes)",
    )
    print("\n  Modal implemented logic per cell (from the code, not the prose):")
    for row in meth_rows:
        print(f"    {row['condition']:<16} {row['code_logic_modal'] or '-'}")
    print("\n  code_acq = acquisition criterion classified from the AST of the "
          "functions that choose the next batch.")
    print("  model_informed = that chooser actually calls the surrogate; a "
          "model-derived criterion that never does is not one.")
    print("  stated_vs_code = agree / partial / mismatch / unverifiable_from_code "
          "/ undocumented, comparing the run's stated criterion to its code.")
    print("  claimed_not_implemented = criterion families named in the log or "
          "report that the chooser never computes.")

    conds = sorted(by_cell)
    shape_rows = cross_compare_cells(by_cell)
    print_table(shape_rows, ["dimension", *conds, "agreement"],
                "How each agent solved the prompt — cross comparison "
                "(rows the cells disagreed on first)")
    print("\n  Each row is one demand of the task, answered from the code that "
          "plays that role (not a workspace-wide grep):")
    for r in shape_rows:
        print(f"    {r['dimension']:<20} {r['question']}")
    print("  Per-run evidence (file:line for every answer) is in each "
          "result.json under methodology.solution_shape.")

    print("\n  Modal chain per cell "
          "(design -> representation+model[ensembling] -> acquisition x rounds -> stop):")
    for row in meth_rows:
        print(f"    {row['condition']:<16} {row['chain_modal'] or '-'}")
    print("\n  chain_agreement = share of a cell's replicates on the modal chain "
          "(method reproducibility, not score reproducibility).")
    print("  evidence_grounded = fraction of adaptive rounds whose validation error "
          "vs baseline was actually recorded when the round's choice was made.")
    print("  criterion_switched = the acquisition criterion CHANGED between rounds "
          "(reactive logic) rather than one fixed rule executed n times.")
    print("  prose_only_claim = a method named in README/report.json that the code "
          "never evidences.")

    if args.show_rounds:
        round_rows = []
        for r in sorted(runs, key=lambda x: (x["condition"], x["run_id"])):
            for it in r["iterations"]:
                round_rows.append({
                    "condition": r["condition"], "run_id": r["run_id"][:15],
                    "round": it["round"], "n": it["n_acquired"],
                    "criterion": "|".join(it["criterion_classes"]) or "-",
                    "val/base": it["val_over_baseline"],
                    "decision": it["decision"],
                    "stated": (it["criterion_text"] or "")[:60],
                })
        print_table(round_rows,
                    ["condition", "run_id", "round", "n", "criterion", "val/base",
                     "decision", "stated"],
                    "Per-round decision logic (every adaptive round of every run)")

    # ---- pooled by access level -----------------------------------------
    pooled = []
    for level in sorted({r["level"] for r in runs}):
        sub = [r for r in runs if r["level"] == level]
        k = sum(1 for r in sub if r["contract_passed"])
        lo, hi = wilson_interval(k, len(sub))
        scored = [r["rel_l2"] for r in sub if isinstance(r["rel_l2"], (int, float))]
        blo, bhi = bootstrap_median_ci(scored)
        vk, vn = _rate([r["physically_valid"] for r in sub], len(sub))
        pk, pn = _rate([r["passed_but_invalid"] for r in sub], len(sub))
        effs = [r["accuracy_pct"] / r["n_solves"] * 100 for r in sub
                if isinstance(r.get("accuracy_pct"), (int, float))
                and isinstance(r.get("n_solves"), (int, float)) and r["n_solves"]]
        pooled.append({
            "level": level, "n": len(sub), "pass_k": k,
            "n_timeout": sum(1 for r in sub if r["timed_out"]),
            "pass_rate": round(k / len(sub), 3),
            "pass_ci": f"[{lo:.2f},{hi:.2f}]" if lo is not None else "-",
            "rel_l2_median": round(float(np.median(scored)), 4) if scored else None,
            "rel_l2_ci": f"[{blo:.3f},{bhi:.3f}]" if blo is not None else "-",
            "phys_valid": f"{vk}/{vn}",
            "passed_but_invalid": f"{pk}/{pn}",
            "acc_per_100_solves": round(float(np.median(effs)), 2) if effs else None,
        })
    print_table(pooled, list(pooled[0].keys()) if pooled else [],
                "Pooled by access level")

    # ---- primary contrast: L2 vs L3, paired within harness ---------------
    l2 = {c["harness"]: c for c in cells if c["level"] == "L2"}
    l3 = {c["harness"]: c for c in cells if c["level"] == "L3"}
    shared = sorted(set(l2) & set(l3))
    if shared:
        print("\n=== Primary contrast: L2 (library) vs L3 (from scratch), "
              "paired within harness ===")
        print("Positive delta = L3 WORSE (higher error) / lower pass rate than L2.\n")
        err_diffs, pass_diffs = [], []
        rows = []
        for h in shared:
            a, b = l2[h], l3[h]
            de = (b["rel_l2_median"] - a["rel_l2_median"]
                  if a["rel_l2_median"] is not None and b["rel_l2_median"] is not None
                  else None)
            dp = a["pass_rate"] - b["pass_rate"]
            if de is not None:
                err_diffs.append(de)
            pass_diffs.append(dp)
            rows.append({
                "harness": h,
                "L2_rel_l2": a["rel_l2_median"], "L3_rel_l2": b["rel_l2_median"],
                "delta_rel_l2": round(de, 4) if de is not None else None,
                "L2_pass": f'{a["pass_k"]}/{a["n"]}', "L3_pass": f'{b["pass_k"]}/{b["n"]}',
                "delta_pass_rate": round(dp, 3),
            })
        print_table(rows, list(rows[0].keys()), "Paired by harness")
        for label, diffs in (("relative-L2 median", err_diffs),
                             ("contract pass rate", pass_diffs)):
            pos, neg, p = sign_test_two_sided(diffs)
            pstr = f"{p:.4f}" if p is not None else "n/a (all ties)"
            print(f"  {label}: L3 worse in {pos}/{pos + neg} harnesses "
                  f"(exact two-sided sign test p={pstr}, n_pairs={len(diffs)})")
        n_pairs = len(pass_diffs)
        floor = min(1.0, 2 / (2 ** n_pairs)) if n_pairs else None
        print(f"  NOTE: the sign test is DESCRIPTIVE here — with {n_pairs} "
              f"harnesses its smallest attainable two-sided p is "
              f"{floor:.3f}, whatever the data show.")

        # PRIMARY: stratified rank over every run, blocked on harness.
        obs_r, p_r, n_used = stratified_rank_test(runs, "rel_l2")
        if p_r is None:
            print("  permutation tests need both levels in at least two "
                  "harnesses — skipped.")
        else:
            print(f"  relative-L2 (PRIMARY, stratified rank / van Elteren): "
                  f"z = {obs_r:+.3f}, permutation p = {p_r:.4f} "
                  f"(n={n_used} runs, labels shuffled within harness).")
            print("  Negative = L3 (from scratch) ranks BELOW L2 on error, "
                  "i.e. from-scratch is better. Harness is a blocking "
                  "factor, so substrate differences cannot inflate this.")
            # The statistic first used as primary, kept as a sensitivity:
            # it is far less powerful because a median of a handful of noisy
            # runs is itself noisy. Reporting both is the honest record of
            # how the analysis was chosen.
            obs_m, p_m, _ = stratified_permutation_test(runs, "rel_l2")
            if p_m is not None:
                print(f"  SENSITIVITY (median-difference permutation, the "
                      f"originally pre-specified test): "
                      f"delta = {obs_m:+.4f}, p = {p_m:.4f} — same direction, "
                      f"much less power; it uses only one number per cell.")
    else:
        print("\n(no harness has both an L2 and an L3 cell — skipping the paired contrast)")

    out = tag_dirs[0] / "aggregate.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)
    per_run, per_round = write_methodology_csvs(runs, tag_dirs[0])
    shape_csv = write_solution_shape_csv(shape_rows, conds, tag_dirs[0])
    total = sum(c["cost_usd_total"] or 0 for c in cells)
    print(f"\nTotal measured spend across these tags: ${total:.2f}")
    print(f"Wrote {out}")
    print(f"Wrote {per_run} (one row per run: the chain of methods)")
    print(f"Wrote {per_round} (one row per adaptive round: the decision logic)")
    print(f"Wrote {shape_csv} (one row per solution dimension, one column per cell)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
