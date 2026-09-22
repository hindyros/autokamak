# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Benchmark CLI: run any task on any harness, validate, compare.

    python -m autotokamak.bench run      --task benchmarks/tasks/L3_from_scratch.yaml \\
                                         --harness claude_sdk [--model ...] [--tag aug09] [--dry-run]
    python -m autotokamak.bench validate --workspace experiments/<tag>/<cond>/<run_id>/workspace \\
                                         --task benchmarks/tasks/L3_from_scratch.yaml
    python -m autotokamak.bench score    --run-dir experiments/<tag>/<cond>/<run_id>
    python -m autotokamak.bench diagnose --tag aug09 [--rescore]
    python -m autotokamak.bench methodology --tag aug09 [--json]
    python -m autotokamak.bench compare  --tag aug09
    python -m autotokamak.bench freeze-testset [--n 60] [--seed 20260809]

``run`` creates experiments/<tag>/<condition>/<run_id>/{workspace/, trace.json,
result.json}, invokes the harness, then runs contract validation and (when the
frozen test set exists) head-to-head scoring.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from autotokamak.agent.runners.config import REPO_ROOT

BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
FROZEN_TESTSET = BENCHMARKS_DIR / "assets" / "test_set.h5"
EVAL_GRID_JSON = BENCHMARKS_DIR / "assets" / "eval_grid.json"
TEST_PARAMS_JSON = BENCHMARKS_DIR / "assets" / "test_params.json"


def _utc_run_id() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def cmd_run(args) -> int:
    from autotokamak.bench.taskspec import TaskSpec
    from autotokamak.harnesses import get_harness

    task = TaskSpec.from_yaml(args.task)
    harness = get_harness(args.harness)
    condition = harness.condition_for(task)
    tag = args.tag or _dt.datetime.now(_dt.UTC).strftime("%Y%m%d")

    run_id = _utc_run_id()
    run_dir = EXPERIMENTS_DIR / tag / condition / run_id
    workspace = run_dir / "workspace"

    if args.dry_run:
        info = harness.dry_run_info(task, workspace, args.model)
        info["run_dir"] = str(run_dir)
        print(json.dumps(info, indent=2))
        return 0

    # 1-second run_id resolution: parallel launches of the same condition
    # must not silently share (and overwrite) one run dir.
    for suffix in range(2, 100):
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            run_dir = EXPERIMENTS_DIR / tag / condition / f"{run_id}-{suffix}"
            workspace = run_dir / "workspace"
    print(f"[bench] condition={condition} run_dir={run_dir}")
    result = harness.run(
        task,
        workspace,
        run_dir=run_dir,
        model=args.model,
        timeout_seconds=args.timeout or task.timeout_seconds,
    )

    payload = result.to_dict()
    payload["task"] = {
        "task_id": task.task_id,
        "prompt_version": task.prompt_version,
        "path": str(task.source_path),
    }
    # Replicate index: campaigns now run each cell n times, and the paper's
    # unit of analysis is the cell, not the run. Recording it here beats
    # parsing it back out of run-id collision suffixes.
    if args.rep is not None:
        payload["replicate"] = int(args.rep)

    from autotokamak.bench.contract import score_against_frozen, validate_deliverables

    contract = validate_deliverables(
        workspace, task,
        grid_json=EVAL_GRID_JSON if EVAL_GRID_JSON.is_file() else None,
        run_predict_check=not args.skip_predict_check,
    )
    payload["contract"] = contract.to_dict()

    # Score whenever the predictor itself works — a missing README must not
    # cost a paid cell its head-to-head number. contract.passed still gates
    # "did it meet the full deliverable contract" in the matrix.
    predict_ok = all(
        contract.gates.get(g) for g in ("predict_runs", "predict_shape", "predict_grid")
    )
    if predict_ok and FROZEN_TESTSET.is_file() and not args.skip_frozen_score:
        try:
            payload["frozen_score"] = score_against_frozen(workspace, FROZEN_TESTSET)
        except Exception as exc:  # noqa: BLE001
            payload["frozen_score"] = {"error": f"{type(exc).__name__}: {exc}"}

    from autotokamak.bench.diagnostics import compute_diagnostics
    from autotokamak.bench.methodology import extract_methodology

    payload["diagnostics"] = compute_diagnostics(
        workspace,
        frozen_score=payload.get("frozen_score"),
        contract_passed=contract.passed,
    )
    # What the agent did and why, beside what it scored. Deterministic and
    # LLM-free, so it costs nothing and re-extracts identically.
    try:
        payload["methodology"] = extract_methodology(workspace)
    except Exception as exc:  # noqa: BLE001 — never lose a paid run to this
        payload["methodology"] = {"error": f"{type(exc).__name__}: {exc}"}

    (run_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    diag = payload["diagnostics"]
    print(f"[bench] status={result.status} contract_passed={contract.passed} "
          f"physically_valid={diag.get('physically_valid')}")
    chain = (payload.get("methodology") or {}).get("chain_signature")
    if chain:
        print(f"[bench] method chain: {chain}")
    if diag.get("passed_gates_but_invalid"):
        print("[bench] WARNING: passed every gate but the prediction is not "
              "physically valid — see result.json diagnostics")
    print(f"[bench] result: {run_dir / 'result.json'}")
    return 0 if result.status == "completed" else 1


def cmd_validate(args) -> int:
    from autotokamak.bench.contract import validate_deliverables
    from autotokamak.bench.taskspec import TaskSpec

    task = TaskSpec.from_yaml(args.task)
    report = validate_deliverables(
        Path(args.workspace), task,
        grid_json=EVAL_GRID_JSON if EVAL_GRID_JSON.is_file() else None,
        run_predict_check=not args.skip_predict_check,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


def cmd_score(args) -> int:
    """Backfill head-to-head scoring for an existing run (no agent re-run)."""
    from autotokamak.bench.contract import score_against_frozen

    if not FROZEN_TESTSET.is_file():
        print(f"No frozen test set: {FROZEN_TESTSET}", file=sys.stderr)
        return 1
    run_dir = Path(args.run_dir) if args.run_dir else None
    workspace = Path(args.workspace) if args.workspace else run_dir / "workspace"
    try:
        score = score_against_frozen(workspace, FROZEN_TESTSET)
    except Exception as exc:  # noqa: BLE001
        score = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(score, indent=2))
    if run_dir and (run_dir / "result.json").is_file():
        payload = json.loads((run_dir / "result.json").read_text())
        payload["frozen_score"] = score
        (run_dir / "result.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[bench] updated {run_dir / 'result.json'}")
    return 0 if "error" not in score else 1


def cmd_diagnose(args) -> int:
    """Backfill the diagnostics block onto runs that predate it.

    ``--rescore`` re-invokes each workspace's ``predict.py`` so the
    structural stats (NaN-mask agreement, prediction spread) that are only
    computable with predictions in hand are added too. That costs solver/CPU
    time but no API spend, so the whole archived corpus can be brought onto
    the current measurement set without paying for re-runs.
    """
    from autotokamak.bench.contract import score_against_frozen
    from autotokamak.bench.diagnostics import compute_diagnostics
    from autotokamak.bench.methodology import extract_methodology

    tag_dir = EXPERIMENTS_DIR / args.tag
    if not tag_dir.is_dir():
        print(f"No such tag dir: {tag_dir}", file=sys.stderr)
        return 1

    n_ok = n_rescored = n_flagged = 0
    for result_path in sorted(tag_dir.glob("*/*/result.json")):
        run_dir = result_path.parent
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {run_dir.name}: unreadable result.json ({exc})")
            continue
        workspace = run_dir / "workspace"

        fs = payload.get("frozen_score")
        # Staleness is keyed on the NEWEST scoring field, so adding a metric
        # to score_against_frozen automatically makes archived runs stale
        # rather than silently leaving them on the old metric set.
        REQUIRED_SCORE_KEYS = ("nan_mask_agreement", "test_rel_l2_full_grid",
                               "exterior_inflation")
        needs = args.rescore and (
            not isinstance(fs, dict)
            or any(k not in fs for k in REQUIRED_SCORE_KEYS)
        )
        if needs and FROZEN_TESTSET.is_file() and (workspace / "predict.py").is_file():
            try:
                payload["frozen_score"] = score_against_frozen(workspace, FROZEN_TESTSET)
                n_rescored += 1
            except Exception as exc:  # noqa: BLE001
                payload["frozen_score"] = {"error": f"{type(exc).__name__}: {exc}"}

        contract_passed = (payload.get("contract") or {}).get("passed")
        payload["diagnostics"] = compute_diagnostics(
            workspace,
            frozen_score=payload.get("frozen_score"),
            contract_passed=contract_passed,
        )
        try:
            payload["methodology"] = extract_methodology(workspace)
        except Exception as exc:  # noqa: BLE001
            payload["methodology"] = {"error": f"{type(exc).__name__}: {exc}"}
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        n_ok += 1
        d = payload["diagnostics"]
        if d.get("passed_gates_but_invalid"):
            n_flagged += 1
            print(f"  FLAG {payload.get('condition', run_dir.parent.name)}/{run_dir.name}: "
                  f"passed gates, physically invalid "
                  f"(rel_l2={d.get('scored_test_rel_l2_mean')}, "
                  f"mask={(payload.get('frozen_score') or {}).get('nan_mask_agreement')})")

    print(f"[bench] diagnosed {n_ok} runs ({n_rescored} rescored, "
          f"{n_flagged} passed-but-invalid)")
    return 0


def cmd_methodology(args) -> int:
    """Extract (or re-extract) the methodology record for a tag's runs.

    Answers the question the score cannot: what chain of methods did each
    agent settle on, and what was its stated reasoning at each adaptive
    round. Deterministic and LLM-free, so it is safe to re-run at any time;
    it rewrites only ``result.json["methodology"]``, never the workspace.
    """
    from autotokamak.bench.methodology import extract_methodology

    tag_dir = EXPERIMENTS_DIR / args.tag
    if not tag_dir.is_dir():
        print(f"No such tag dir: {tag_dir}", file=sys.stderr)
        return 1

    n = 0
    for result_path in sorted(tag_dir.glob("*/*/result.json")):
        run_dir = result_path.parent
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {run_dir.name}: unreadable result.json ({exc})")
            continue
        try:
            meth = extract_methodology(run_dir / "workspace")
        except Exception as exc:  # noqa: BLE001
            meth = {"error": f"{type(exc).__name__}: {exc}"}
        payload["methodology"] = meth
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        n += 1

        cond = payload.get("condition", run_dir.parent.name)
        if args.json:
            print(json.dumps({"condition": cond, "run_id": run_dir.name, **meth}, indent=2))
            continue
        logic = meth.get("decision_logic") or {}
        print(f"\n{cond}/{run_dir.name}")
        print(f"  chain   : {meth.get('chain_signature')}")
        print(f"  rounds  : {logic.get('n_rounds')} "
              f"(criterion switched between rounds: {logic.get('criterion_switched')}, "
              f"evidence-grounded: {logic.get('evidence_grounded_fraction')})")
        for it in meth.get("iterations") or []:
            print(f"    r{it['round']}: n={it['n_acquired']} "
                  f"[{','.join(it['criterion_classes']) or '-'}] "
                  f"val/base={it['val_over_baseline']} -> {it['decision']}")
            if it.get("criterion_text"):
                print(f"        \"{it['criterion_text'][:110]}\"")
        prose_only = (meth.get("evidence") or {}).get("prose_only_terms") or {}
        if prose_only:
            print(f"  claimed in prose but absent from the code: {prose_only}")
    print(f"\n[bench] methodology extracted for {n} runs")
    return 0


def cmd_compare(args) -> int:
    from autotokamak.bench.report import collect_results, render_table

    tag_dir = EXPERIMENTS_DIR / args.tag
    if not tag_dir.is_dir():
        print(f"No such tag dir: {tag_dir}", file=sys.stderr)
        return 1
    print(render_table(collect_results(tag_dir)))
    return 0


def cmd_freeze_testset(args) -> int:
    """Solve the frozen test parameters once; write benchmarks/assets/test_set.h5.

    Ground truth for head-to-head scoring across every condition. Parameters
    come from test_params.json when present (committed, seeded), else are
    drawn uniformly here with the recorded seed.
    """
    import numpy as np

    if FROZEN_TESTSET.is_file() and not args.force:
        print(f"Already exists (use --force to regenerate): {FROZEN_TESTSET}")
        return 0

    bounds = {
        "r0": (0.35, 0.55),
        "a": (0.10, 0.20),
        "kappa": (1.0, 1.6),
        "delta": (0.0, 0.4),
        "Ip": (80e3, 200e3),
    }
    if TEST_PARAMS_JSON.is_file():
        records = json.loads(TEST_PARAMS_JSON.read_text())
        print(f"Using {len(records)} committed test parameters: {TEST_PARAMS_JSON}")
    else:
        rng = np.random.default_rng(args.seed)
        records = [
            {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in bounds.items()}
            for _ in range(args.n)
        ]
        TEST_PARAMS_JSON.parent.mkdir(parents=True, exist_ok=True)
        TEST_PARAMS_JSON.write_text(json.dumps(records, indent=2))
        print(f"Drew {len(records)} params (seed={args.seed}) → {TEST_PARAMS_JSON}")

    from autotokamak.bench.freeze import solve_testset

    n_ok = solve_testset(records, FROZEN_TESTSET, grid_json=EVAL_GRID_JSON)
    print(f"Solved {n_ok}/{len(records)} → {FROZEN_TESTSET}")
    return 0 if n_ok > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m autotokamak.bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(
        dest="cmd",
        metavar="<run|validate|score|diagnose|methodology|compare|freeze-testset>",
    )
    sub.required = True

    p = sub.add_parser("run", help="Run a task on a harness")
    p.add_argument("--task", required=True, help="Path to benchmarks/tasks/<task>.yaml")
    p.add_argument("--harness", required=True, help="echo|ursa|dspy|claude_sdk|pi|cursor")
    p.add_argument("--model", default=None, help="Model override for this run")
    p.add_argument("--tag", default=None, help="experiments/<tag>/ bucket (default: UTC date)")
    p.add_argument("--timeout", type=int, default=None, help="Override task timeout_seconds")
    p.add_argument("--dry-run", action="store_true", help="Print what would run; no agent call")
    p.add_argument("--skip-predict-check", action="store_true")
    p.add_argument("--skip-frozen-score", action="store_true")
    p.add_argument("--rep", type=int, default=None,
                   help="Replicate index for this cell, recorded in result.json")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("validate", help="Contract-validate an existing workspace")
    p.add_argument("--workspace", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--skip-predict-check", action="store_true")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("score", help="(Re)score an existing workspace on the frozen test set")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="experiments/<tag>/<cond>/<run_id>; updates its result.json")
    g.add_argument("--workspace", help="score a bare workspace; print only")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("diagnose", help="Backfill the diagnostics block onto a tag's runs")
    p.add_argument("--tag", required=True)
    p.add_argument("--rescore", action="store_true",
                   help="Re-run predict.py so structural stats are computed too")
    p.set_defaults(fn=cmd_diagnose)

    p = sub.add_parser("methodology",
                       help="Extract each run's method chain and per-round decision logic")
    p.add_argument("--tag", required=True)
    p.add_argument("--json", action="store_true", help="Full records instead of a summary")
    p.set_defaults(fn=cmd_methodology)

    p = sub.add_parser("compare", help="Comparison table across a tag's runs")
    p.add_argument("--tag", required=True)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("freeze-testset", help="Build the frozen head-to-head test set")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_freeze_testset)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
