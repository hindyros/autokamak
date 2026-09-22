# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Harness ABC + RunResult — the contract every agent-substrate adapter meets."""
from __future__ import annotations

import signal
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from autotokamak.bench.taskspec import TaskSpec


class HarnessTimeout(BaseException):
    """Raised by :func:`time_limit` when an in-process adapter overruns.

    Inherits **BaseException**, not Exception, for the same reason
    KeyboardInterrupt does: agent frameworks wrap their step loops in broad
    ``except Exception`` handlers and will otherwise swallow the timeout and
    carry on. That is not hypothetical — it is exactly what happened in the
    first n=10 campaign attempt, where URSA absorbed the alarm and ran for
    over 10 hours against a 90-minute cap. ``signal.alarm`` is one-shot, so
    a single swallowed alarm disables the cap permanently.
    """


@contextmanager
def time_limit(seconds: int | None):
    """Wall-clock cap for adapters whose engine is an in-process call.

    The subprocess-jailed adapters (claude_sdk/pi/cursor) get their cap from
    asyncio/subprocess directly. URSA and DSPy call their engines in-process,
    so they need SIGALRM: it raises :class:`HarnessTimeout` in the main thread
    at the next bytecode boundary.

    Without this, ``timeout_seconds`` was accepted and silently ignored by
    both adapters, and they ran 3-6x the wall budget of every other harness
    in matrix-v2 (33-98 min vs 8-13 min) — an unequal-compute confound that
    contaminates any cross-harness comparison.

    Limits, stated plainly: SIGALRM is Unix-only and main-thread-only, and it
    does not reap grandchild processes the agent may have spawned (the same
    caveat applies to the asyncio cap in claude_sdk). Where it cannot arm, it
    yields unguarded rather than failing the run.
    """
    if (
        not seconds
        or seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    # Belt and braces: re-arm on every fire. If something downstream still
    # manages to absorb the raise, the next alarm lands 30s later rather
    # than never — the cap degrades to "late", not "gone".
    def _fire(signum, frame):  # noqa: ANN001 - signal handler signature
        signal.alarm(30)
        raise HarnessTimeout(f"exceeded timeout_seconds={seconds}")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class RunResult:
    status: Literal["completed", "errored", "interrupted", "timeout"]
    run_id: str
    condition: str          # "<level>-<harness>", e.g. "L3-claude_sdk"
    harness: str
    model: str
    workspace: Path
    trace_path: Path | None
    wall_seconds: float
    cost_usd: float | None = None   # filled where the substrate reports it
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "condition": self.condition,
            "harness": self.harness,
            "model": self.model,
            "workspace": str(self.workspace),
            "trace_path": str(self.trace_path) if self.trace_path else None,
            "wall_seconds": round(self.wall_seconds, 1),
            "cost_usd": self.cost_usd,
            "error": self.error,
            **({"extra": self.extra} if self.extra else {}),
        }


class Harness(ABC):
    """One agent substrate.

    Responsibilities of every adapter:
      * materialize ``task.symlinks`` in the workspace,
      * write a ``bench.trace`` RunTrace under ``run_dir`` (schema v1 plus
        the additive ``harness`` prompt field),
      * never write outside ``workspace`` and ``run_dir``,
      * map substrate-native events into trace rounds/steps best-effort and
        keep any raw event stream as ``run_dir/<name>_events.jsonl``.
    """

    name: ClassVar[str]

    @abstractmethod
    def run(
        self,
        task: TaskSpec,
        workspace: Path,
        *,
        run_dir: Path,
        model: str | None = None,
        timeout_seconds: int | None = None,
    ) -> RunResult:
        """Execute ``task`` in ``workspace``; artifacts/trace under ``run_dir``."""

    # ---- shared helpers -------------------------------------------------

    def condition_for(self, task: TaskSpec) -> str:
        return f"{task.access_level}-{self.name}"

    def resolve_model(self, task: TaskSpec, model: str | None) -> str | None:
        return model or task.model_for(self.name)

    def prepare_workspace(self, task: TaskSpec, workspace: Path) -> None:
        from autotokamak.agent.runners.config import materialize_symlinks

        workspace.mkdir(parents=True, exist_ok=True)
        materialize_symlinks(workspace, task.symlinks)

    def workspace_note(self, workspace: Path) -> str:
        """Cwd-jail warning appended to the prompt by every subprocess-jailed
        adapter (claude_sdk/pi/cursor) — identical wording across harnesses so
        no cell gets private help against the shared cwd-escape failure mode."""
        return (
            f"\n\nRUNTIME NOTE: your workspace directory is\n{Path(workspace).resolve()}\n"
            "Create ALL files under this directory (absolute paths are "
            "safest). Never write outside it, and if you change "
            "directory for exploration, change back before writing."
        )

    def dry_run_info(self, task: TaskSpec, workspace: Path,
                     model: str | None = None) -> dict[str, Any]:
        """What would run — subclasses override to expose exact argv/env keys."""
        return {
            "harness": self.name,
            "condition": self.condition_for(task),
            "model": self.resolve_model(task, model),
            "workspace": str(workspace),
            "timeout_seconds": task.timeout_seconds,
        }
