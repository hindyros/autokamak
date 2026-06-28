"""Action dispatchers invoked by the meta-loop runner.

Three actions, three dispatchers. Each takes ``(payload, state)`` and
returns a serializable dict the next iteration's diagnostics can consume.

The ``extend_search`` dispatcher is the only one that triggers a nested LLM
call — it programmatically invokes
``agent.runners.plan_execute_feedback.main`` against an overlay prompt
that adds the meta-agent's focus directive to the existing Phase-2
problem text.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from autotokamak.agent.orchestrator.schema import (
    ActionDecision,
    ExtendSearchFocus,
    RegenDatasetOverrides,
    TerminateReason,
)
from autotokamak.data.schema import SweepConfig
from autotokamak.data.sweep import run_sweep


@dataclass
class MetaState:
    """Mutable state threaded across iterations.

    Holds the live dataset path, the best surrogate report so far, the
    sweep config used for any regen action, and a running RMSE history.
    """

    workspace: Path
    current_dataset_h5: Path
    base_sweep_config: Optional[SweepConfig] = None
    best_winner_payload: Optional[dict] = None
    best_winner_path: Optional[Path] = None
    best_surrogate_report: Optional[dict] = None
    best_rmse: float = float("inf")
    rmse_history: list[float] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    phase2_prompt: Path = Path(
        "src/autotokamak/agent/prompts/surrogate_automl.yaml"
    )
    seed: int = 0

    def relative_dataset(self) -> str:
        return str(self.current_dataset_h5)


# ----------------------------- regen_dataset ----------------------------- #

def _deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def regen_dataset(payload: RegenDatasetOverrides, state: MetaState) -> Dict[str, Any]:
    """Apply overrides to the base sweep config and run the deterministic sweep.

    No LLM involved. The new dataset replaces ``state.current_dataset_h5``.
    """
    if state.base_sweep_config is None:
        raise RuntimeError(
            "regen_dataset requires meta_config.base_sweep_config to be set"
        )
    raw = state.base_sweep_config.model_dump(mode="json")
    for k, v in payload.overrides.items():
        _deep_set(raw, k, v)
    new_cfg = SweepConfig.model_validate(raw)

    datasets_dir = state.workspace / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    iter_idx = len(state.actions_taken)
    new_cfg = new_cfg.model_copy(update={"output_path": f"iter{iter_idx}_dataset.h5"})

    result = run_sweep(new_cfg, datasets_dir)
    # Persist the merged config alongside the dataset so the trace can show
    # exactly what was regenerated.
    cfg_path = datasets_dir / f"iter{iter_idx}_config.yaml"
    cfg_path.write_text(yaml.safe_dump(new_cfg.model_dump(mode="json"), sort_keys=False))

    # The new dataset becomes the current one.
    state.current_dataset_h5 = Path(result.dataset_path)
    state.base_sweep_config = new_cfg

    return {
        "kind": "regen_dataset",
        "dataset_path": result.dataset_path,
        "config_path": str(cfg_path),
        "n_requested": result.n_requested,
        "n_succeeded": result.n_succeeded,
        "n_isoflux_used": result.n_isoflux_used,
        "config_hash": result.config_hash,
        "overrides_applied": dict(payload.overrides),
        "rationale": payload.rationale,
    }


# ----------------------------- extend_search ----------------------------- #

def _build_overlay_prompt(
    phase2_prompt_path: Path,
    focus: ExtendSearchFocus,
    dataset_path: Path,
    workspace: Path,
) -> Path:
    """Write a copy of the Phase-2 prompt with a 'FOCUS' block injected.

    The overlay prompt has the same structure as ``surrogate_automl.yaml``
    plus a short ``FOCUS DIRECTIVE`` section the nested LLM reads. The
    overlay's ``workspace`` is the meta-iteration's sub-workspace.
    """
    raw = yaml.safe_load(phase2_prompt_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Phase-2 prompt {phase2_prompt_path} must be a YAML mapping")

    focus_lines = ["", "FOCUS DIRECTIVE FROM META-AGENT", ""]
    if focus.models_to_emphasize:
        focus_lines.append(
            "  Emphasize these models: " + ", ".join(focus.models_to_emphasize)
        )
    if focus.widen_params:
        focus_lines.append(
            "  Widen the search range for: " + ", ".join(focus.widen_params)
        )
    if focus.n_trials_hint is not None:
        focus_lines.append(
            f"  Suggested total trial budget: {focus.n_trials_hint}"
        )
    if focus.rationale:
        focus_lines.append(f"  Reason: {focus.rationale}")

    raw["problem"] = (raw.get("problem", "") or "") + "\n" + "\n".join(focus_lines)
    raw["workspace"] = str(workspace)
    # Override the dataset symlink so the nested run sees the current
    # meta-state dataset.
    symlinks = list(raw.get("symlinks", []) or [])
    symlinks = [
        s for s in symlinks
        if not (isinstance(s, dict) and s.get("dest") == "dataset.h5")
    ]
    symlinks.append({"source": str(dataset_path), "dest": "dataset.h5"})
    raw["symlinks"] = symlinks

    overlay_path = workspace / "overlay_prompt.yaml"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return overlay_path


def extend_search(payload: ExtendSearchFocus, state: MetaState) -> Dict[str, Any]:
    """Invoke ``plan_execute_feedback`` as a sub-LLM run on the Phase-2 prompt.

    Loads the nested winner.pkl + report.json afterward, updates
    ``state.best_*`` if the new RMSE beats the prior best.
    """
    iter_idx = len(state.actions_taken)
    sub_ws = state.workspace / "surrogate_runs" / f"iter{iter_idx}"
    sub_ws.mkdir(parents=True, exist_ok=True)

    overlay_path = _build_overlay_prompt(
        phase2_prompt_path=state.phase2_prompt,
        focus=payload,
        dataset_path=state.current_dataset_h5,
        workspace=sub_ws,
    )

    # Programmatic invocation. Imported lazily so the orchestrator module
    # has no hard dependency on langchain/ursa at import time.
    from agent.runners.plan_execute_feedback import main as feedback_main

    started = time.time()
    feedback_main(
        config_path=str(overlay_path),
        cli_model=None,
        workspace_override=str(sub_ws),
        trace_enabled=True,
        experiments_dir=state.workspace / "experiments",
    )
    elapsed = time.time() - started

    # Load nested artifacts and update best.
    winner_path = sub_ws / "outputs" / "winner.pkl"
    report_path = sub_ws / "outputs" / "report.json"
    nested_rmse: Optional[float] = None
    if winner_path.is_file() and report_path.is_file():
        import joblib

        nested_winner = joblib.load(winner_path)
        nested_report = json.loads(report_path.read_text())
        nested_rmse = float(nested_report.get("val_psi_rmse", float("inf")))
        if nested_rmse < state.best_rmse:
            state.best_rmse = nested_rmse
            state.best_winner_payload = nested_winner
            state.best_winner_path = winner_path
            state.best_surrogate_report = nested_report

    return {
        "kind": "extend_search",
        "overlay_prompt": str(overlay_path),
        "sub_workspace": str(sub_ws),
        "elapsed_seconds": elapsed,
        "nested_val_rmse": nested_rmse,
        "winner_path": str(winner_path) if winner_path.is_file() else None,
        "models_emphasized": list(payload.models_to_emphasize),
        "widen_params": list(payload.widen_params),
        "rationale": payload.rationale,
    }


# ----------------------------- terminate ----------------------------- #

def terminate(payload: TerminateReason, state: MetaState) -> Dict[str, Any]:
    return {
        "kind": "terminate",
        "reason": payload.reason,
        "confidence": payload.confidence,
    }


# ----------------------------- dispatch ----------------------------- #

DISPATCH = {
    "regen_dataset": regen_dataset,
    "extend_search": extend_search,
    "terminate": terminate,
}


def dispatch(decision: ActionDecision, state: MetaState) -> Dict[str, Any]:
    payload = decision.selected_payload()
    if payload is None:
        raise ValueError(
            f"ActionDecision.action={decision.action} but the corresponding payload is None"
        )
    handler = DISPATCH[decision.action]
    return handler(payload, state)


__all__ = [
    "DISPATCH",
    "MetaState",
    "dispatch",
    "extend_search",
    "regen_dataset",
    "terminate",
]
