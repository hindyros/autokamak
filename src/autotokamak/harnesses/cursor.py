# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Cursor CLI harness (``cursor-agent`` headless mode).

    cursor-agent -p "<prompt>" --output-format stream-json --trust --force [--model M]

``--trust --force`` are REQUIRED for a guaranteed non-interactive run
(``-p`` alone is known to hang waiting for trust confirmation); a hard
subprocess timeout backstops it. The raw stream-json is kept at
``run_dir/cursor_events.jsonl``.

Known limitations (accepted, documented in benchmarks/README.md): no tool
allowlist, and the agent may index the enclosing git repo — the L3
no-autotokamak rule therefore rests on the prompt plus the contract's
import audit, the same trust model as URSA's unrestricted Bash.

Auth: CURSOR_API_KEY env, or a prior interactive ``cursor-agent login``.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from autotokamak.bench.taskspec import TaskSpec
from autotokamak.bench.trace import RunTrace, utc_run_id
from autotokamak.harnesses.base import Harness, RunResult

CURSOR_BIN = "cursor-agent"


def _argv(prompt: str, model: str | None) -> list[str]:
    argv = [CURSOR_BIN, "-p", prompt, "--output-format", "stream-json",
            "--trust", "--force"]
    if model:
        argv += ["--model", model]
    return argv


class CursorHarness(Harness):
    """The ``cursor-agent`` command-line coding agent, as a subprocess.

    Takes bare model identifiers (``gpt-5.2``) rather than the provider-prefixed
    form the other adapters expect, which is why the campaign pins models per
    adapter in the task file instead of passing one override.

    Consumes its vendor's own quota rather than a provider API key, so its
    dollar figures throughout this project are estimates derived from token
    counts and are labelled as such.
    """

    name = "cursor"

    def dry_run_info(self, task: TaskSpec, workspace: Path,
                     model: str | None = None) -> dict[str, Any]:
        info = super().dry_run_info(task, workspace, model)
        info.update({
            "argv": _argv("<prompt>", self.resolve_model(task, model)),
            "cwd": str(workspace),
            "env_keys": ["CURSOR_API_KEY (or prior `cursor-agent login`)"],
        })
        return info

    def run(
        self,
        task: TaskSpec,
        workspace: Path,
        *,
        run_dir: Path,
        model: str | None = None,
        timeout_seconds: int | None = None,
    ) -> RunResult:
        """Run one agent against ``task`` inside ``workspace``.

        Implementations must materialise ``task.symlinks``, write a
        :class:`~autotokamak.bench.trace.RunTrace` under ``run_dir``, confine
        all writes to ``workspace`` and ``run_dir``, honour
        ``timeout_seconds``, and leave the substrate's raw event stream at
        ``run_dir/<name>_events.jsonl``. See ``CONTRIBUTING.md``.
        """
        started = time.time()
        run_id = utc_run_id()
        model_name = self.resolve_model(task, model) or "(cursor default)"
        timeout = timeout_seconds or task.timeout_seconds
        self.prepare_workspace(task, workspace)

        trace = RunTrace.open_at(
            run_dir,
            prompt_path=task.source_path or Path("<inline>"),
            model=model_name,
            workspace=str(workspace),
            harness=self.name,
            run_id=run_id,
        )
        round_rec = trace.start_round(1)
        trace.record_plan_steps(round_rec, ["cursor-agent session (agent plans internally)"])
        step = trace.start_step(round_rec, 1, "cursor-agent -p session")

        events_path = run_dir / "cursor_events.jsonl"
        status, error, tail = "completed", None, ""
        usage, result_evt = None, None
        try:
            # Own process group + killpg, not subprocess.run(timeout=...):
            # that kills only the direct child, then blocks in communicate()
            # until grandchildren release the inherited pipes. See the note
            # in pi.py — the identical pattern deadlocked a pi cell for 9h.
            # It matters more here: this adapter already documents a known
            # cursor-agent "-p hang mode", so the timeout path is load-bearing.
            proc = subprocess.Popen(
                _argv(task.render_prompt(self.name) + self.workspace_note(workspace),
                      self.resolve_model(task, model)),
                cwd=workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                out, err = proc.communicate(timeout=timeout)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                out, err = proc.communicate()
                status = "timeout"
                error = f"exceeded {timeout}s (known -p hang mode — killed)"
                returncode = None

            if out:
                events_path.write_text(out, encoding="utf-8")
                tail = _events_tail(out)
                usage, result_evt = _final_result(out)
            if err:
                (run_dir / "cursor_stderr.log").write_text(err, encoding="utf-8")
            if returncode is None:
                pass  # already marked timeout above
            elif returncode != 0:
                # Prefer the stream's own verdict (parity with claude_sdk,
                # which keys off ResultMessage.subtype): a non-zero exit
                # after a successful final result event is a shutdown
                # artifact, not an agent failure.
                if result_evt and not result_evt.get("is_error") \
                        and result_evt.get("subtype") == "success":
                    error = None
                else:
                    status = "errored"
                    error = f"cursor-agent exit {returncode}: {(err or '')[-500:]}"
            elif result_evt and result_evt.get("is_error"):
                status = "errored"
                error = f"cursor result is_error (subtype={result_evt.get('subtype')})"
        except FileNotFoundError:
            status, error = "errored", f"'{CURSOR_BIN}' not found on PATH"
        except KeyboardInterrupt:
            status, error = "interrupted", "KeyboardInterrupt"

        trace.finish_step(step, ok=(status == "completed"),
                          result_text=tail or (error or ""))
        trace.record_artifacts(workspace, expected_artifacts=task.expected_artifacts)
        if status == "completed":
            trace.mark_completed()
        elif status == "interrupted":
            trace.mark_interrupted()
        else:
            trace.mark_errored(RuntimeError(error or status))

        return RunResult(
            status=status,
            run_id=run_id,
            condition=self.condition_for(task),
            harness=self.name,
            model=model_name,
            workspace=workspace,
            trace_path=trace._path,
            wall_seconds=time.time() - started,
            error=error,
            # cursor reports token usage but no dollar figure; cost is derived
            # downstream (tools/cost_report.py) from a model price table.
            extra={"usage": usage} if usage else {},
        )


def _final_result(stdout: str) -> tuple[dict | None, dict | None]:
    """Token usage + the final ``type=="result"`` event from the stream."""
    usage, result_evt = None, None
    for ln in stdout.splitlines():
        try:
            evt = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "result":
            result_evt = evt
            if isinstance(evt.get("usage"), dict):
                usage = evt["usage"]
    return usage, result_evt


def _events_tail(stdout: str, n: int = 5) -> str:
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()][-n:]
    out = []
    for ln in lines:
        try:
            evt = json.loads(ln)
            out.append(str(evt.get("type") or "event") + ": " + str(evt)[:300])
        except json.JSONDecodeError:
            out.append(ln[:300])
    return "\n".join(out)
