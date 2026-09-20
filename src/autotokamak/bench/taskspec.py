# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""TaskSpec — the harness-agnostic definition of one benchmark task.

A task YAML under ``benchmarks/tasks/`` declares WHAT the agent must do
(the problem text, the deliverable contract, the access level); a harness
adapter decides HOW an agent substrate executes it. One task therefore runs
unchanged across every harness, which is what makes conditions comparable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TaskSpec(BaseModel):
    """One benchmark task, loaded from YAML and hashed into every run record.

    A task is the frozen contract between the benchmark and whatever is being
    measured: the problem text the agent is given verbatim, the artifacts it
    must produce, the access level (which decides whether importing the
    platform library is allowed and audited), and the per-substrate model pin.

    The text is deliberately immutable in practice. Changing it changes what is
    being measured, so ``prompt_version`` exists to make that visible and the
    aggregation refuses to pool across versions. ``TaskSpec.from_yaml`` records
    the file's SHA-256 for the same reason.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    # L2 = may import autotokamak; L3 = from scratch, import audited & forbidden.
    access_level: Literal["L2", "L3"]
    problem: str
    expected_artifacts: list[str] = Field(default_factory=list)
    # Dotted "<module>:<func>" consumed by bench.scoring.try_score.
    scorer: str | None = None
    scorer_kwargs: dict[str, Any] = Field(default_factory=dict)
    # [{"source": "./OpenFUSIONToolkit", "dest": "OpenFUSIONToolkit"}, ...]
    symlinks: list[dict[str, str]] = Field(default_factory=list)
    # Per-harness default model, e.g. {"ursa": "openai:gpt-5.2",
    # "claude_sdk": "claude-sonnet-4-5"}. "default" is the fallback key.
    model: dict[str, str] = Field(default_factory=dict)
    # Per-harness prompt appendix (e.g. URSA's graph_store.sqlite note).
    harness_notes: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=21600, ge=60)
    # Consumed by harnesses that support outer feedback/fix loops.
    feedback_rounds: int = Field(default=1, ge=1, le=5)
    # Bump whenever the problem text changes. Runs are only comparable within
    # one version; traces also record the YAML's sha256.
    prompt_version: int = Field(default=1, ge=1)

    # Set by from_yaml; lets harnesses resolve relative paths.
    source_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskSpec:
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Task YAML must be a mapping: {path}")
        raw.setdefault("task_id", path.stem)
        spec = cls.model_validate(raw)
        spec.source_path = path.resolve()
        return spec

    def render_prompt(self, harness: str) -> str:
        """Problem text plus the harness-specific appendix, if any."""
        note = self.harness_notes.get(harness, "").strip()
        return self.problem if not note else f"{self.problem.rstrip()}\n\n{note}\n"

    def model_for(self, harness: str) -> str | None:
        return self.model.get(harness) or self.model.get("default")
