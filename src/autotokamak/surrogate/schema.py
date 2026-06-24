"""Pydantic v2 schemas for surrogate AutoML configs and outputs.

Mirrors the pattern in ``autotokamak.core.schema``:

- ``SearchSpec`` — what the agent emits per outer-loop round (which models,
  which hyperparameter ranges, how many trials each)
- ``StudyResult`` — what ``automl.run_study`` returns and what
  ``automl.summarize_study`` reads back
- ``SurrogateConfig`` — top-level ``surrogate_config.yaml`` validated at
  runtime; lets the agent (and the scorer) trust required fields are present
- ``SurrogateReport`` — schema for the workspace's ``outputs/report.json``;
  the scorer's ``report_parseable`` hard gate validates the agent's output
  against this exact shape
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


ModelKind = Literal["gp", "kernel_ridge", "poly_ridge", "mlp"]
MODEL_KINDS: tuple[ModelKind, ...] = ("gp", "kernel_ridge", "poly_ridge", "mlp")


# ----------------------------- search spec ----------------------------- #

class ParamRange(BaseModel):
    """One hyperparameter's Optuna sampler spec.

    The agent emits these in ``SearchSpec.per_model_search_space``. We keep the
    shape simple: type-tagged dicts that mirror Optuna's ``suggest_*`` methods.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["float", "int", "categorical", "loguniform"]
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    step: Optional[float] = None


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: ModelKind
    n_trials: int = Field(ge=1, le=200)
    search_space: Dict[str, ParamRange]


class SearchSpec(BaseModel):
    """One outer-loop round of decisions the agent emits.

    Stored under ``outputs/search_history.jsonl`` (one line per round) so the
    scorer can measure ``search_efficiency`` and ``agent_decisiveness``.
    """

    model_config = ConfigDict(extra="forbid")
    round: int = Field(ge=1)
    models: List[ModelSpec] = Field(min_length=1)
    n_pca_components: int = Field(ge=1, le=64)
    val_metric: Literal["psi_rmse"] = "psi_rmse"
    action: Literal["initial", "widen_range", "add_model", "tighten_around_best", "terminate"]
    rationale: str = ""


# ----------------------------- study result ----------------------------- #

class TrialRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    number: int
    value: float
    params: Dict[str, Any]


class ModelStudyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: ModelKind
    n_trials: int
    best_value: float
    best_params: Dict[str, Any]
    edge_hit: Dict[str, bool] = Field(default_factory=dict)
    trials: List[TrialRecord] = Field(default_factory=list)


class StudyResult(BaseModel):
    """Aggregate of all per-model studies run in one ``run_study`` call."""

    model_config = ConfigDict(extra="forbid")
    spec: SearchSpec
    per_model: List[ModelStudyResult]
    storage_path: str

    @property
    def best_overall(self) -> ModelStudyResult:
        return min(self.per_model, key=lambda m: m.best_value)


# ----------------------------- workspace config ----------------------------- #

class SurrogateConfig(BaseModel):
    """``surrogate_config.yaml`` written by the agent into the workspace.

    Holds run-wide settings the agent does NOT iterate on per outer-loop round
    (dataset path, time budget, RNG seed). The per-round search decisions live
    in the emitted ``search_spec.json`` instead.
    """

    model_config = ConfigDict(extra="allow")
    dataset_h5: str = Field(description="Relative path to the Phase-1 dataset.h5.")
    time_budget_seconds: int = Field(default=300, ge=10, le=3600)
    n_pca_components_default: int = Field(default=12, ge=1, le=64)
    seed: int = 0
    k_folds: int = Field(default=4, ge=2, le=8)
    test_frac: float = Field(default=2 / 16, gt=0.0, lt=0.5)
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SurrogateConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError("Top-level YAML must be a mapping/object.")
        return cls.model_validate(raw)


# ----------------------------- final report ----------------------------- #

class SurrogateReport(BaseModel):
    """Final ``outputs/report.json`` the agent writes; scorer validates this.

    Quality terms read these fields verbatim; missing or out-of-range values
    fail the ``report_parseable`` hard gate.
    """

    model_config = ConfigDict(extra="allow")
    winner_model_name: ModelKind
    winner_hyperparams: Dict[str, Any]
    val_psi_rmse: float = Field(ge=0.0)
    test_psi_rmse: float = Field(ge=0.0)
    baseline_mean_psi_rmse: float = Field(ge=0.0)
    pca_n_components: int = Field(ge=1)
    pca_explained_var: float = Field(ge=0.0, le=1.0)
    n_total_trials: int = Field(ge=1)
    n_outer_rounds: int = Field(ge=1)
    terminated_by: Literal["agent", "rounds_cap"] = "agent"
    models_tried: List[ModelKind]


__all__ = [
    "MODEL_KINDS",
    "ModelKind",
    "ModelSpec",
    "ModelStudyResult",
    "ParamRange",
    "SearchSpec",
    "StudyResult",
    "SurrogateConfig",
    "SurrogateReport",
    "TrialRecord",
]
