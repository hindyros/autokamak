"""Pydantic schemas for the meta-agent: config, decisions, per-iteration records.

The LLM's structured output schema is ``ActionDecision`` — the meta-loop forces
the model to emit one of these per iteration, so the action space is small,
typed, and validated before dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


ActionKind = Literal["regen_dataset", "extend_search", "terminate"]


class RegenDatasetOverrides(BaseModel):
    """Flat overrides applied on top of the current sweep config.

    Use dotted keys (``"sampling.n_samples"``, ``"fixed.mesh_dx"``) so the
    LLM can target any field without us hard-coding which knobs are tweakable.
    Validation against ``SweepConfig`` happens in the dispatcher after the
    merge.
    """

    model_config = ConfigDict(extra="allow")
    overrides: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ExtendSearchFocus(BaseModel):
    """What the meta-agent wants the nested Phase-2 sub-run to emphasize.

    The dispatcher writes a thin overlay prompt that prepends this focus
    block to the existing ``surrogate_automl.yaml`` problem text. The nested
    LLM is otherwise unchanged.
    """

    model_config = ConfigDict(extra="forbid")
    models_to_emphasize: List[Literal["gp", "kernel_ridge", "poly_ridge", "mlp"]] = Field(
        default_factory=list,
        description="Subset of the zoo to focus on; empty = let Phase-2 pick.",
    )
    widen_params: List[str] = Field(
        default_factory=list,
        description="Specific hyperparameters to widen (e.g. 'gp.length_scale').",
    )
    n_trials_hint: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Total trials suggestion for the nested Phase-2 search.",
    )
    rationale: str = ""


class TerminateReason(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str
    confidence: Literal["low", "medium", "high"] = "medium"


class ActionDecision(BaseModel):
    """The typed JSON the meta-agent's LLM emits per iteration.

    Exactly one of (``regen``, ``extend``, ``terminate``) is populated; the
    ``action`` discriminator selects which.
    """

    model_config = ConfigDict(extra="forbid")
    action: ActionKind
    regen: Optional[RegenDatasetOverrides] = None
    extend: Optional[ExtendSearchFocus] = None
    terminate: Optional[TerminateReason] = None
    diagnosis: str = Field(
        default="",
        description="Free-text statement of what the agent thinks the bottleneck is.",
    )

    def selected_payload(self) -> BaseModel | None:
        return {"regen_dataset": self.regen,
                "extend_search": self.extend,
                "terminate": self.terminate}[self.action]


class MetaConfig(BaseModel):
    """Top-level ``surrogate_meta.yaml`` schema.

    Differs from ``SurrogateConfig`` because the meta-agent's job is
    cross-phase orchestration, not surrogate training per se.
    """

    model_config = ConfigDict(extra="allow")
    max_iterations: int = Field(default=3, ge=1, le=20)
    initial_dataset_h5: str = Field(description="Path to the starting dataset.")
    base_sweep_config: Optional[str] = Field(
        default=None,
        description="Path to a SweepConfig YAML the regen_dataset action overrides.",
    )
    phase2_prompt: str = Field(
        default="src/autotokamak/agent/prompts/surrogate_automl.yaml",
        description="The prompt the extend_search action invokes (codegen mode only).",
    )
    phase2_mode: Literal["structured", "codegen"] = Field(
        default="structured",
        description="How extend_search runs Phase-2: 'structured' = deterministic "
        "automl_loop with one typed LLM decision per round; 'codegen' = the "
        "legacy nested plan_execute_feedback agent.",
    )
    phase2_max_rounds: int = Field(default=3, ge=1, le=10)
    holdout_test_frac: float = Field(
        default=0.15,
        gt=0.0,
        lt=0.5,
        description="Fraction of the initial dataset's successful samples frozen "
        "into the held-out test shard at meta-loop start.",
    )
    holdout_min_test: int = Field(default=2, ge=1)
    # Early-stopping quality bar (both optional; loop stops when EITHER is
    # met by the frozen-shard RMSE; max_iterations remains the safety net
    # for unreachable targets):
    target_rmse: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Absolute shard-RMSE target in the dataset's psi units.",
    )
    target_rmse_ratio: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Relative target: stop when shard RMSE <= ratio * baseline "
        "(mean-predictor) RMSE. Scale-free, so it survives dataset changes.",
    )
    seed: int = 0
    workspace: str = "examples/surrogate_meta"
    model: str = "openai:gpt-5.2"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MetaConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("Top-level YAML must be a mapping/object.")
        return cls.model_validate(raw)


class MetaIterationRecord(BaseModel):
    """One row in the meta-trace's iteration log."""

    model_config = ConfigDict(extra="allow")
    iteration: int
    started_utc: str
    finished_utc: str = ""
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    decision: ActionDecision
    result: Dict[str, Any] = Field(default_factory=dict)
    rmse_after: Optional[float] = None
    parent_run_id: Optional[str] = None  # for extend_search actions only


class MetaReport(BaseModel):
    """Final report.json written into the meta workspace."""

    model_config = ConfigDict(extra="allow")
    n_iterations: int
    terminated_by: Literal["agent", "iterations_cap", "target_reached"]
    # The resolved absolute target the run stopped against (None = no target).
    target_rmse: Optional[float] = None
    initial_rmse: Optional[float] = None
    # None = no winner was ever produced. All RMSEs (final, baseline, and the
    # per-iteration rmse_history) are measured on the SAME frozen test shard.
    final_rmse: Optional[float] = Field(default=None, ge=0.0)
    baseline_rmse: float = Field(ge=0.0)
    test_shard_path: Optional[str] = None
    n_test_samples: Optional[int] = None
    n_train_pool_samples: Optional[int] = None
    winner_model_name: str
    winner_hyperparams: Dict[str, Any]
    rmse_history: List[float] = Field(default_factory=list)
    actions_taken: List[ActionKind] = Field(default_factory=list)


__all__ = [
    "ActionDecision",
    "ActionKind",
    "ExtendSearchFocus",
    "MetaConfig",
    "MetaIterationRecord",
    "MetaReport",
    "RegenDatasetOverrides",
    "TerminateReason",
]
