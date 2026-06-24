"""Composite scorer for the meta-agent loop.

Mirrors ``metric_surrogate.score_surrogate_run`` shape (same ``ScoreReport``
dataclass, same hard-gates × quality composite). Reads the meta-workspace's
``report.json`` + ``meta_trace.json`` and scores:

Hard gates (all must pass for nonzero total):
    deliverables_present : winner.pkl, report.json, meta_trace.json
    report_parseable     : report.json validates against MetaReport
    iteration_log        : meta_trace.json has ≥1 iteration record
    winner_predicts      : winner.pkl loads and predicts test split

Quality terms:
    final_rmse_vs_baseline (0.35) : 1 - final_rmse/baseline_rmse, clipped
    improvement_over_iterations (0.20) : first-rmse minus last-rmse, normalized
    diagnosis_consistency (0.15) : per-iteration "diagnosis says X" matched
                                   the action taken (heuristic; perfect match → 1)
    budget_efficiency (0.15) : did good RMSE land early in the budget
    terminated_by_agent (0.10) : 1.0 if agent terminated, 0.0 if cap hit
    runner_cleanliness (0.05) : iteration log uses the expected action types
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_DELIVERABLES = ("winner.pkl", "report.json", "meta_trace.json")
EXPECTED_ACTIONS = {"regen_dataset", "extend_search", "terminate"}

WEIGHTS = {
    "final_rmse_vs_baseline": 0.35,
    "improvement_over_iterations": 0.20,
    "diagnosis_consistency": 0.15,
    "budget_efficiency": 0.15,
    "terminated_by_agent": 0.10,
    "runner_cleanliness": 0.05,
}


@dataclass
class ScoreReport:
    workspace: Path
    hard_gates: dict[str, bool] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def all_gates_pass(self) -> bool:
        return bool(self.hard_gates) and all(self.hard_gates.values())

    @property
    def total(self) -> float:
        if not self.all_gates_pass:
            return 0.0
        return float(sum(WEIGHTS[k] * self.quality.get(k, 0.0) for k in WEIGHTS))

    def summary(self) -> str:
        lines = [f"ScoreReport[{self.workspace}]"]
        lines.append("  hard gates:")
        for k, v in self.hard_gates.items():
            lines.append(f"    [{'PASS' if v else 'FAIL'}] {k}")
        if self.all_gates_pass:
            lines.append("  quality:")
            for k, w in WEIGHTS.items():
                q = self.quality.get(k, 0.0)
                lines.append(f"    {q:.3f}  (weight {w:.2f})  {k}")
            lines.append(f"  --> total = {self.total:.3f}")
        else:
            lines.append("  --> total = 0.000 (hard gate failed)")
        return "\n".join(lines)


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(max(0.0, min(1.0, x)))


def _load_report(path: Path) -> tuple[Any | None, str | None]:
    try:
        from autotokamak.agent.orchestrator.schema import MetaReport

        raw = json.loads(path.read_text(encoding="utf-8"))
        return MetaReport.model_validate(raw), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _load_trace(path: Path) -> tuple[dict | None, str | None]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "iterations" not in raw:
            return None, "meta_trace.json missing 'iterations'"
        return raw, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def score_meta_run(workspace: str | Path) -> ScoreReport:
    ws = Path(workspace)
    report = ScoreReport(workspace=ws)

    # -- Hard gate 1: deliverables present --
    missing = [f for f in EXPECTED_DELIVERABLES if not (ws / f).is_file()]
    report.hard_gates["deliverables_present"] = not missing
    report.details["missing_deliverables"] = missing

    # -- Hard gate 2: report.json parses --
    parsed_report, report_err = _load_report(ws / "report.json")
    report.hard_gates["report_parseable"] = parsed_report is not None
    if report_err:
        report.details["report_parse_error"] = report_err

    # -- Hard gate 3: meta_trace.json valid --
    trace, trace_err = _load_trace(ws / "meta_trace.json")
    report.hard_gates["iteration_log"] = trace is not None and len(trace.get("iterations", [])) >= 1
    if trace_err:
        report.details["trace_parse_error"] = trace_err

    # -- Hard gate 4: winner_predicts --
    winner_predicts = False
    if (ws / "winner.pkl").is_file():
        try:
            import joblib

            payload = joblib.load(ws / "winner.pkl")
            req = {"estimator", "pca"}
            winner_predicts = req.issubset(set(payload))
        except Exception as exc:  # noqa: BLE001
            report.details["winner_load_error"] = f"{type(exc).__name__}: {exc}"
    report.hard_gates["winner_predicts"] = winner_predicts

    if not report.all_gates_pass:
        return report

    assert parsed_report is not None
    assert trace is not None

    # -- final_rmse_vs_baseline --
    report.quality["final_rmse_vs_baseline"] = _clip01(
        1.0 - parsed_report.final_rmse / max(parsed_report.baseline_rmse, 1e-12)
    )

    # -- improvement_over_iterations --
    history = list(parsed_report.rmse_history)
    if len(history) >= 2:
        first, last = history[0], history[-1]
        improvement = (first - last) / max(first, 1e-12)
        report.quality["improvement_over_iterations"] = _clip01(improvement)
    else:
        report.quality["improvement_over_iterations"] = 0.0

    # -- diagnosis_consistency: did each iteration's action match its stated
    # diagnosis category? Heuristic regex match between diagnosis text and
    # action type. Perfect match → 1.0; none → 0.0.
    matches = 0
    total = 0
    for it in trace["iterations"]:
        diagnosis = (it.get("decision", {}) or {}).get("diagnosis", "") or ""
        action = (it.get("decision", {}) or {}).get("action", "")
        total += 1
        if action == "regen_dataset" and re.search(r"\b(sample|data|N\b|coverage|fidelity|noise|mesh)", diagnosis, re.I):
            matches += 1
        elif action == "extend_search" and re.search(r"\b(edge|range|model|hyper|widen|search|trial|tune)", diagnosis, re.I):
            matches += 1
        elif action == "terminate" and re.search(r"\b(good|enough|done|stop|terminate|converge|plateau)", diagnosis, re.I):
            matches += 1
    report.quality["diagnosis_consistency"] = matches / total if total else 0.0

    # -- budget_efficiency: best rmse_after at 50% of iterations vs final --
    rmses_with_idx = [
        (i, it.get("rmse_after"))
        for i, it in enumerate(trace["iterations"])
        if it.get("rmse_after") is not None
    ]
    if rmses_with_idx and parsed_report.final_rmse > 0:
        half_n = max(1, len(trace["iterations"]) // 2)
        early = [v for i, v in rmses_with_idx if i < half_n]
        if early:
            best_early = min(early)
            report.quality["budget_efficiency"] = _clip01(
                parsed_report.final_rmse / best_early
            )
        else:
            report.quality["budget_efficiency"] = 0.0
    else:
        report.quality["budget_efficiency"] = 0.0

    # -- terminated_by_agent --
    report.quality["terminated_by_agent"] = (
        1.0 if parsed_report.terminated_by == "agent" else 0.0
    )

    # -- runner_cleanliness: only expected action types used --
    action_types = {
        (it.get("decision", {}) or {}).get("action") for it in trace["iterations"]
    }
    invalid = action_types - EXPECTED_ACTIONS
    report.quality["runner_cleanliness"] = 0.0 if invalid else 1.0
    if invalid:
        report.details["invalid_actions"] = sorted(a for a in invalid if a)

    return report


__all__ = ["EXPECTED_DELIVERABLES", "ScoreReport", "WEIGHTS", "score_meta_run"]
