#!/usr/bin/env python
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Guards that stand between a campaign and a wasted budget.

A replicated campaign is the most expensive thing this repo does: tens of
paid agent sessions, hours of wall clock, and nothing recoverable if it goes
wrong in the wrong way. The failures that actually cost money are not exotic:

  * a crash or Ctrl-C at hour 18, and the re-run repeats every finished cell
    because ``bench run`` always mints a new run_id (``completed``);
  * a harness that should not be in the campaign at all — this paper is
    pinned to OpenAI models, and ``claude_sdk`` is Anthropic by construction
    (``preflight``);
  * a missing key, a logged-out CLI or a full disk discovered at 2am, after
    the cells have already burned their timeout (``preflight``);
  * spend running away while nobody is watching (``spend``, used by the
    driver to stop the campaign at a ceiling).

Each subcommand is exit-code-driven so ``tools/run_campaign.sh`` can use it
as a gate.

    python tools/campaign_guard.py preflight --harnesses "ursa dspy pi cursor" \
        --task benchmarks/tasks/L3_mini_v3.yaml [--min-disk-gb 10]
    python tools/campaign_guard.py completed --tag t --condition L3-pi --rep 2
    python tools/campaign_guard.py spend --tag t [--budget 150]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO_ROOT / "experiments"

# Substrate -> (env var that must be set, CLI that must be on PATH).
HARNESS_REQUIREMENTS: dict[str, tuple[str | None, str | None]] = {
    "ursa": ("OPENAI_API_KEY", None),
    "dspy": ("OPENAI_API_KEY", None),
    "pi": ("OPENAI_API_KEY", "pi"),
    "cursor": (None, "cursor-agent"),          # CURSOR_API_KEY or a CLI login
    "claude_sdk": ("ANTHROPIC_API_KEY", "claude"),
    "echo": (None, None),
}

# Harnesses that cannot honour an OpenAI model pin. Running one silently
# confounds the harness axis with the model axis AND spends on another
# provider, so the driver refuses unless told otherwise.
NON_OPENAI = {"claude_sdk"}


def _runs(tag_dir: Path):
    for rp in sorted(tag_dir.glob("*/*/result.json")):
        try:
            yield rp, json.loads(rp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue


def cmd_completed(args) -> int:
    """Exit 0 when this (condition, replicate) already finished. For --resume."""
    tag_dir = EXPERIMENTS / args.tag
    if not tag_dir.is_dir():
        return 1
    for _rp, r in _runs(tag_dir):
        if r.get("condition") != args.condition:
            continue
        if args.rep is not None and r.get("replicate") != args.rep:
            continue
        # A timed-out or errored cell is NOT done: re-running it is the point
        # of a resume. Only a completed run counts.
        if r.get("status") == "completed":
            print(f"{args.condition} rep {args.rep}: already completed "
                  f"({_rp.parent.name})")
            return 0
    return 1


def measured_spend(tag: str) -> tuple[float, list[str]]:
    """Dollars spent under a tag, best-effort, with what could not be priced.

    Self-reported cost first; then tools/cost_report.py's derived figures
    (cursor reports tokens, not dollars); then URSA's own per-call accounting
    for runs the watchdog killed before they could write a result.json — the
    aborted n=10 attempt hid $12.31 of real spend that way.
    """
    tag_dir = EXPERIMENTS / tag
    total, unpriced = 0.0, []
    if not tag_dir.is_dir():
        return 0.0, ["no such tag dir"]

    priced_runs = set()
    for rp, r in _runs(tag_dir):
        run_key = (r.get("condition"), rp.parent.name)
        c = r.get("cost_usd")
        if isinstance(c, (int, float)):
            total += float(c)
            priced_runs.add(run_key)
        else:
            unpriced.append(f"{run_key[0]}/{run_key[1]}")

    csv_path = tag_dir / "cost_report.csv"
    if csv_path.is_file():
        import csv as _csv

        try:
            with csv_path.open(newline="") as fh:
                for row in _csv.DictReader(fh):
                    key = (row.get("condition"), row.get("run_id"))
                    if key in priced_runs:
                        continue
                    try:
                        total += float(row["cost_usd"])
                        priced_runs.add(key)
                        if f"{key[0]}/{key[1]}" in unpriced:
                            unpriced.remove(f"{key[0]}/{key[1]}")
                    except (KeyError, TypeError, ValueError):
                        continue
        except OSError:
            pass

    # Killed runs: no result.json, but URSA wrote its own accounting.
    for metrics in tag_dir.glob("*/*/workspace/ursa_metrics/*.json"):
        key = (metrics.parents[2].parent.name, metrics.parents[2].name)
        if key in priced_runs:
            continue
        try:
            data = json.loads(metrics.read_text(encoding="utf-8"))
            c = (data.get("costs") or {}).get("total_usd")
            if isinstance(c, (int, float)):
                total += float(c)
                priced_runs.add(key)
        except Exception:  # noqa: BLE001
            continue
    return total, unpriced


def cmd_reconcile(args) -> int:
    """Give killed cells a result.json so they survive into the analysis.

    A cell the driver's watchdog kills never writes one: the Python process
    dies mid-run. The run directory and its workspace remain, but every
    reporting path in this repo globs for ``result.json``, so the cell
    silently vanishes — the campaign's denominator shrinks and a substrate
    that always times out looks like a substrate that was never run. That is
    the difference between "ursa completed 0 of 8" and "ursa is absent from
    the table".

    The stub records what is knowable from the filesystem and nothing more;
    it is marked so it can never be mistaken for a measured run.
    """
    tag_dir = EXPERIMENTS / args.tag
    if not tag_dir.is_dir():
        print(f"no such tag dir: {tag_dir}", file=sys.stderr)
        return 1

    written = 0
    for run_dir in sorted(tag_dir.glob("*/*")):
        if not run_dir.is_dir() or (run_dir / "result.json").is_file():
            continue
        if not (run_dir / "workspace").is_dir() and not (run_dir / "trace.json").is_file():
            continue
        condition = run_dir.parent.name
        payload = {
            "condition": condition,
            "run_id": run_dir.name,
            "harness": condition.partition("-")[2],
            "status": "killed",
            "error": ("no result.json: the process was killed before it could "
                      "write one (driver watchdog, OOM, or an operator stop)"),
            "reconciled_by": "tools/campaign_guard.py reconcile",
            "measured": False,
        }
        trace_path = run_dir / "trace.json"
        if trace_path.is_file():
            try:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                for key in ("model", "replicate", "started_utc", "task"):
                    if key in trace:
                        payload[key] = trace[key]
            except Exception:  # noqa: BLE001
                pass
        if args.dry_run:
            print(f"would write stub: {run_dir}/result.json")
        else:
            (run_dir / "result.json").write_text(json.dumps(payload, indent=2),
                                                 encoding="utf-8")
        written += 1
    print(f"[reconcile] {written} killed cell(s) "
          f"{'would be' if args.dry_run else ''} recorded")
    return 0


def cmd_spend(args) -> int:
    total, unpriced = measured_spend(args.tag)
    print(f"{total:.2f}")
    if args.verbose and unpriced:
        print(f"  unpriced runs (cost not yet recoverable): {len(unpriced)}",
              file=sys.stderr)
    if args.budget is not None and total >= args.budget:
        print(f"[guard] spend ${total:.2f} has reached the ${args.budget:.2f} "
              f"ceiling", file=sys.stderr)
        return 2
    return 0


def cmd_preflight(args) -> int:
    """Everything that is cheaper to discover now than at 2am."""
    problems, notes = [], []
    harnesses = [h.split(":")[0] for h in args.harnesses.split()]

    for h in harnesses:
        if h in NON_OPENAI and not args.allow_non_openai:
            problems.append(
                f"harness '{h}' cannot run an OpenAI model pin — it would spend "
                f"on another provider and confound the harness axis with the "
                f"model axis. Drop it, or pass --allow-non-openai deliberately.")
        env, cli = HARNESS_REQUIREMENTS.get(h, (None, None))
        if env and not os.environ.get(env):
            problems.append(f"harness '{h}' needs ${env}, which is not set "
                            f"(is .env loaded?)")
        if cli and not shutil.which(cli):
            problems.append(f"harness '{h}' needs '{cli}' on PATH — not found")
        if h not in HARNESS_REQUIREMENTS:
            problems.append(f"unknown harness '{h}'")

    if args.task:
        task = Path(args.task)
        if not task.is_file():
            problems.append(f"task file missing: {task}")
        else:
            try:
                from autotokamak.bench.taskspec import TaskSpec

                spec = TaskSpec.from_yaml(task)
                pinned = set(spec.model or {})
                missing_pin = [h for h in harnesses
                               if h not in pinned and h not in {"echo"}]
                if missing_pin:
                    problems.append(
                        f"task {task.name} pins models for {sorted(pinned)} but "
                        f"not for {missing_pin} — those cells would run the "
                        f"adapter's default model, which is not comparable")
                notes.append(f"task {spec.task_id} prompt_version={spec.prompt_version} "
                             f"timeout={spec.timeout_seconds}s "
                             f"feedback_rounds={spec.feedback_rounds}")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"task file will not load: {exc}")

    for asset in ("test_set.h5", "eval_grid.json", "test_params.json"):
        p = REPO_ROOT / "benchmarks" / "assets" / asset
        if not p.is_file():
            problems.append(f"frozen asset missing: benchmarks/assets/{asset} "
                            f"— runs would be unscoreable")

    free_gb = shutil.disk_usage(REPO_ROOT).free / 2**30
    if free_gb < args.min_disk_gb:
        problems.append(
            f"only {free_gb:.1f} GB free; a campaign writes ~20-110 MB of "
            f"datasets per run and a full disk fails solves mid-flight "
            f"(need >= {args.min_disk_gb} GB)")
    else:
        notes.append(f"disk free: {free_gb:.1f} GB")

    # A dirty tree means the archived runs cannot be tied to a commit.
    try:
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=30).stdout
        src_dirty = [ln for ln in dirty.splitlines()
                     if ln[3:].startswith(("src/", "benchmarks/tasks/",
                                           "tools/run_campaign.sh"))]
        if src_dirty:
            notes.append(f"WARNING: {len(src_dirty)} uncommitted change(s) under "
                         f"src/ or benchmarks/tasks/ — commit first so the "
                         f"campaign is tied to a revision")
        else:
            head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
            notes.append(f"HEAD {head}, no uncommitted src/task changes")
    except Exception:  # noqa: BLE001
        pass

    for n in notes:
        print(f"[preflight] {n}")
    for p in problems:
        print(f"[preflight] BLOCKER: {p}", file=sys.stderr)
    if problems:
        print(f"[preflight] {len(problems)} blocker(s) — not spending anything.",
              file=sys.stderr)
        return 1
    print("[preflight] all checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight", help="Check everything before spending")
    p.add_argument("--harnesses", required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--min-disk-gb", type=float, default=10.0)
    p.add_argument("--allow-non-openai", action="store_true")
    p.set_defaults(fn=cmd_preflight)

    p = sub.add_parser("completed", help="Exit 0 if this cell already finished")
    p.add_argument("--tag", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--rep", type=int, default=None)
    p.set_defaults(fn=cmd_completed)

    p = sub.add_parser("reconcile",
                       help="Write a stub result.json for cells killed mid-run")
    p.add_argument("--tag", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_reconcile)

    p = sub.add_parser("spend", help="Dollars spent so far under a tag")
    p.add_argument("--tag", required=True)
    p.add_argument("--budget", type=float, default=None,
                   help="Exit 2 when spend has reached this ceiling")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_spend)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.exit(main())
