"""Run the full autotokamak agentic pipeline: Phase-1 dataset → Phase-2 surrogate.

Usage:
    python tools/run_full_pipeline.py
    python tools/run_full_pipeline.py --model openai:gpt-5-mini
    python tools/run_full_pipeline.py --regen-dataset       # force Phase-1 even if dataset exists
    python tools/run_full_pipeline.py --skip-phase2         # only rebuild the dataset
    python tools/run_full_pipeline.py --skip-eval           # skip surrogate eval plots
    python tools/run_full_pipeline.py --skip-report         # skip HTML report regen

Chain:
    1. Phase-1 (agentic): plan_execute_feedback on dataset_generation.yaml
       Skipped if outputs/dataset.h5 already exists, unless --regen-dataset.
    2. Phase-2 (agentic): plan_execute_feedback on surrogate_automl.yaml.
    3. eval_surrogate.py against the Phase-2 workspace.
    4. trace_to_html.py to regenerate the browsable report.

Each phase's stdout+stderr streams to a timestamped file in logs/ AND to this
process's stdout, so you can watch progress live and still have a permanent
transcript for the HTML viewer.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_PROMPT = REPO_ROOT / "src/autotokamak/agent/prompts/dataset_generation.yaml"
AUTOML_PROMPT = REPO_ROOT / "src/autotokamak/agent/prompts/surrogate_automl.yaml"
META_PROMPT = REPO_ROOT / "src/autotokamak/agent/prompts/surrogate_meta.yaml"
DATASET_OUT = REPO_ROOT / "examples/dataset_generation/outputs/dataset.h5"
SURROGATE_WORKSPACE = REPO_ROOT / "examples/surrogate_automl"
META_WORKSPACE = REPO_ROOT / "examples/surrogate_meta"
LOGS_DIR = REPO_ROOT / "logs"


def _timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _stream_run(cmd: list[str], log_path: Path, *, env: dict | None = None) -> int:
    """Run cmd; tee stdout+stderr to log_path AND this process's stdout."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"→ running: {' '.join(cmd)}")
    print(f"  log: {log_path.relative_to(REPO_ROOT)}")
    with open(log_path, "wb") as sink:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            sink.write(line)
        return proc.wait()


def _venv_python() -> str:
    """Return the venv's Python if present, else the current interpreter."""
    venv_py = REPO_ROOT / "venv/bin/python"
    return str(venv_py) if venv_py.exists() else sys.executable


def _agent_env() -> dict:
    """Env for the agent runner: PYTHONPATH points at src/autotokamak."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src/autotokamak")
    return env


def phase1_dataset(model: str | None, force: bool) -> tuple[bool, Path | None]:
    """Return (ran, log_path). If skipped, ran=False and log_path=None."""
    if DATASET_OUT.exists() and not force:
        print(f"✓ Phase-1 skipped — dataset already exists at {DATASET_OUT.relative_to(REPO_ROOT)}")
        print(f"  use --regen-dataset to force a fresh agent-driven regeneration")
        return False, None
    if force and DATASET_OUT.exists():
        print(f"! --regen-dataset given; existing {DATASET_OUT.name} will be overwritten by the agent")

    print("\n=========================================================")
    print("PHASE 1 — Agentic dataset generation")
    print("=========================================================")
    log_path = LOGS_DIR / f"dataset_gen_{_timestamp()}.log"
    cmd = [
        _venv_python(), "-u", "-m", "agent.runners.plan_execute_feedback",
        "--config", str(DATASET_PROMPT.relative_to(REPO_ROOT)),
    ]
    if model:
        cmd += ["--model", model]
    rc = _stream_run(cmd, log_path, env=_agent_env())
    if rc != 0:
        print(f"✗ Phase-1 exited with code {rc}", file=sys.stderr)
        return True, log_path
    if not DATASET_OUT.exists():
        print(f"✗ Phase-1 completed but {DATASET_OUT} was not produced", file=sys.stderr)
        return True, log_path
    print(f"✓ Phase-1 complete — dataset at {DATASET_OUT.relative_to(REPO_ROOT)}")
    return True, log_path


def phase2_surrogate(model: str | None) -> tuple[bool, Path]:
    print("\n=========================================================")
    print("PHASE 2 — Agentic surrogate AutoML")
    print("=========================================================")
    if SURROGATE_WORKSPACE.exists():
        print(f"! Removing prior Phase-2 workspace: {SURROGATE_WORKSPACE.relative_to(REPO_ROOT)}")
        import shutil
        shutil.rmtree(SURROGATE_WORKSPACE)
    log_path = LOGS_DIR / f"surrogate_automl_{_timestamp()}.log"
    cmd = [
        _venv_python(), "-u", "-m", "agent.runners.plan_execute_feedback",
        "--config", str(AUTOML_PROMPT.relative_to(REPO_ROOT)),
    ]
    if model:
        cmd += ["--model", model]
    rc = _stream_run(cmd, log_path, env=_agent_env())
    ok = rc == 0 and (SURROGATE_WORKSPACE / "outputs/winner.pkl").exists()
    if not ok:
        print(f"✗ Phase-2 exited with code {rc}; winner.pkl "
              f"{'exists' if (SURROGATE_WORKSPACE / 'outputs/winner.pkl').exists() else 'MISSING'}",
              file=sys.stderr)
    else:
        print(f"✓ Phase-2 complete — workspace at {SURROGATE_WORKSPACE.relative_to(REPO_ROOT)}")
    return ok, log_path


def phase2_meta_loop(model: str | None) -> tuple[bool, Path, Path | None]:
    """Run the Phase-3 meta-loop instead of vanilla Phase-2.

    Returns (ok, log_path, best_sub_workspace).

    The meta-loop wraps Phase-2 and gives the LLM an ``ActionDecision`` each
    iteration: ``extend_search`` (run another Phase-2), ``regen_dataset`` (build
    a bigger dataset via the sweep config), or ``terminate``. Best winner is
    copied into <META_WORKSPACE>/winner.pkl.
    """
    print("\n=========================================================")
    print("PHASE 2 (meta-loop) — Agent chooses: extend_search vs. regen_dataset")
    print("=========================================================")
    if META_WORKSPACE.exists():
        print(f"! Removing prior meta workspace: {META_WORKSPACE.relative_to(REPO_ROOT)}")
        import shutil
        shutil.rmtree(META_WORKSPACE)
    log_path = LOGS_DIR / f"surrogate_meta_{_timestamp()}.log"
    cmd = [
        _venv_python(), "-u", "-m", "agent.runners.meta_loop",
        "--config", str(META_PROMPT.relative_to(REPO_ROOT)),
    ]
    if model:
        cmd += ["--model", model]
    rc = _stream_run(cmd, log_path, env=_agent_env())

    best_sub = _find_best_sub_workspace(META_WORKSPACE)
    ok = rc == 0 and best_sub is not None and (best_sub / "outputs/winner.pkl").is_file()
    if not ok:
        print(f"✗ Meta-loop exited with code {rc}; best sub-workspace "
              f"{'not found' if best_sub is None else best_sub}", file=sys.stderr)
    else:
        print(f"✓ Meta-loop complete — best sub-workspace at "
              f"{best_sub.relative_to(REPO_ROOT)}")
    return ok, log_path, best_sub


def _find_best_sub_workspace(meta_ws: Path) -> Path | None:
    """Return the sub-workspace path (surrogate_runs/iterN) that produced the
    lowest val_psi_rmse in the meta run. Falls back to the newest iterN dir
    if no report.json is parseable."""
    runs_dir = meta_ws / "surrogate_runs"
    if not runs_dir.is_dir():
        return None
    best_path: Path | None = None
    best_rmse = float("inf")
    for sub in sorted(runs_dir.iterdir()):
        report = sub / "outputs/report.json"
        if not report.is_file():
            continue
        try:
            data = json.loads(report.read_text())
            val = float(data.get("val_psi_rmse", float("inf")))
        except Exception:
            continue
        if val < best_rmse:
            best_rmse = val
            best_path = sub
    if best_path is None:
        # Fallback: newest iterN with a winner.pkl.
        for sub in sorted(runs_dir.iterdir(), reverse=True):
            if (sub / "outputs/winner.pkl").is_file():
                return sub
    return best_path


def phase3_eval(workspace: Path = SURROGATE_WORKSPACE) -> bool:
    print("\n=========================================================")
    print(f"Post-run — Surrogate evaluation plots  ({workspace.relative_to(REPO_ROOT)})")
    print("=========================================================")
    cmd = [_venv_python(), str(REPO_ROOT / "tools/eval_surrogate.py"),
           "--workspace", str(workspace)]
    rc = subprocess.call(cmd, cwd=REPO_ROOT)
    if rc != 0:
        print(f"! eval_surrogate.py exited with code {rc}", file=sys.stderr)
        return False
    return True


def phase3_report() -> bool:
    print("\n=========================================================")
    print("Post-run — HTML report regeneration")
    print("=========================================================")
    cmd = [_venv_python(), str(REPO_ROOT / "tools/trace_to_html.py")]
    rc = subprocess.call(cmd, cwd=REPO_ROOT)
    if rc != 0:
        print(f"! trace_to_html.py exited with code {rc}", file=sys.stderr)
        return False
    print(f"  open experiments/_report/index.html")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None,
                    help="LLM override (e.g. openai:gpt-5-mini). Applies to both phases.")
    ap.add_argument("--regen-dataset", action="store_true",
                    help="Force Phase-1 even if outputs/dataset.h5 exists.")
    ap.add_argument("--skip-phase2", action="store_true",
                    help="Stop after Phase-1 (don't run the surrogate AutoML).")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Skip the eval_surrogate.py plots step.")
    ap.add_argument("--skip-report", action="store_true",
                    help="Skip the trace_to_html.py report regen step.")
    ap.add_argument("--enable-meta", action="store_true",
                    help="Use the Phase-3 meta-loop instead of vanilla Phase-2. "
                         "The agent can then autonomously choose to regen the "
                         "dataset (bigger N) if diagnostics say the surrogate "
                         "is sample-bottlenecked.")
    args = ap.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    ran_phase1, _p1_log = phase1_dataset(args.model, args.regen_dataset)
    _ = ran_phase1  # only carried for possible future use

    if args.skip_phase2:
        print("\n(--skip-phase2 given; stopping after Phase-1)")
        return

    if not DATASET_OUT.exists():
        print(f"\n✗ Cannot start Phase-2 — no dataset at {DATASET_OUT}", file=sys.stderr)
        sys.exit(2)

    if args.enable_meta:
        ok, _p2_log, best_sub = phase2_meta_loop(args.model)
        eval_workspace = best_sub if best_sub else META_WORKSPACE
    else:
        ok, _p2_log = phase2_surrogate(args.model)
        eval_workspace = SURROGATE_WORKSPACE
    if not ok:
        print("\n✗ Phase-2 did not produce a winner; skipping eval and report")
        sys.exit(3)

    if not args.skip_eval:
        phase3_eval(eval_workspace)
    if not args.skip_report:
        phase3_report()

    print("\n=========================================================")
    print("PIPELINE DONE")
    print("=========================================================")


if __name__ == "__main__":
    main()
