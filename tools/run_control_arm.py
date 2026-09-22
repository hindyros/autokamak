# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Active-learning control arm: does adaptive sampling actually earn its solves?

Runs the meta-loop twice over, with the action FORCED, so the two arms
differ only in how new solver calls are chosen:

    treatment  forced enrich_active  (residual-UCB, targets measured weakness)
    control    forced regen_dataset  (blind space-filling append)

Both arms get the same per-iteration solve budget, the same seed dataset,
the same Phase-2 search, and the same envelope eval set. The headline metric
is WORST-CELL accuracy per expensive solve: an adaptive method should pay
off precisely where the surrogate is weakest, and the aggregate can hide
that.

This is the experiment docs/active_learning_design.md D5 and the system
paper §5.7 both promise and neither has ever run. Note the prior is NOT
favourable: the one L0 meta-run that exercised enrich_active got worse
(7.73e-4 -> 8.15e-4, refit_became_best: false). A clean negative result here
is a publishable finding, not a failure.

Usage:
    python tools/run_control_arm.py --reps 5
    python tools/run_control_arm.py --reps 1 --iterations 2   # quick shakedown

Costs solver/CPU time; the forced picker makes NO LLM calls, so API spend is
zero.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "benchmarks" / "configs" / "control_arm_meta.yaml"
ARMS = {"active": "forced_enrich", "random": "forced_regen"}


def _seeded_config(config: Path, seed: int, out_dir: Path) -> Path:
    """A copy of the config with its seed replaced.

    Replicates need different seeds or they are not replicates. The L0 path
    is deterministic by design ("same seed => same decisions"), and
    meta_loop.run exposes no seed override, so the seed has to travel in the
    config. Without this, --reps 3 ran one experiment three times and
    produced results identical to fourteen decimal places — observed, on the
    first attempt at this arm.
    """
    import yaml

    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["seed"] = seed
    dest = out_dir / "configs" / f"{config.stem}_seed{seed}.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return dest


def run_one(arm: str, rep: int, config: Path, n_new: int,
            iterations: int | None, out_dir: Path, seed: int) -> dict:
    from autotokamak.agent.runners import meta_loop
    from autotokamak.policies import get_meta_policy

    config = _seeded_config(config, seed, out_dir)
    picker = get_meta_policy(ARMS[arm], forced_n_new=n_new)
    started = time.time()
    report = meta_loop.run(
        str(config),
        pick_action=picker,
        max_iterations_override=iterations,
        enrich_n_new_override=n_new,
        n_samples_override=n_new,          # keep the blind arm's batch equal
        workspace_override=str(out_dir / f"{arm}_rep{rep}"),
    )
    d = report.model_dump() if hasattr(report, "model_dump") else dict(report)
    return {
        "arm": arm,
        "rep": rep,
        "seed": seed,
        "wall_s": round(time.time() - started, 1),
        "n_iterations": d.get("n_iterations"),
        "actions": d.get("actions_taken"),
        "eval_mode": d.get("eval_mode"),
        "baseline_rmse": d.get("baseline_rmse"),
        "initial_rmse": d.get("initial_rmse"),
        "final_rmse": d.get("final_rmse"),
        "rmse_history": d.get("rmse_history"),
        "accuracy_pct": d.get("final_accuracy_pct"),
        "worst_cell_accuracy_pct": d.get("final_worst_cell_accuracy_pct"),
        # Train-pool size is the solver spend: both arms start from the same
        # seed dataset, so the growth IS the budget consumed.
        "n_train_pool_samples": d.get("n_train_pool_samples"),
        "n_test_samples": d.get("n_test_samples"),
        "winner": d.get("winner_model_name"),
        "terminated_by": d.get("terminated_by"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-new", type=int, default=200,
                    help="Solver calls per data iteration — IDENTICAL for both arms")
    ap.add_argument("--iterations", type=int, default=None,
                    help="Override max_iterations (default: from the config)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--base-seed", type=int, default=100,
                    help="Rep r runs with seed base+r; both arms of a rep "
                         "share a seed, so they are paired")
    ap.add_argument("--tag", default="control-arm")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "experiments" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for rep in range(1, args.reps + 1):
        for arm in ARMS:
            print(f"\n=== {arm} arm, rep {rep}/{args.reps} ===", flush=True)
            try:
                row = run_one(arm, rep, Path(args.config), args.n_new,
                              args.iterations, out_dir,
                              seed=args.base_seed + rep)
            except Exception as exc:  # noqa: BLE001 — one bad rep must not lose the rest
                row = {"arm": arm, "rep": rep, "error": f"{type(exc).__name__}: {exc}"}
                print(f"  FAILED: {row['error']}", file=sys.stderr)
            rows.append(row)
            (out_dir / "control_arm_results.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8")
            print(f"  {json.dumps(row)}", flush=True)

    print("\n=== SUMMARY: worst-cell accuracy %, by arm ===")
    for arm in ARMS:
        vals = [r["worst_cell_accuracy_pct"] for r in rows
                if r.get("arm") == arm and isinstance(r.get("worst_cell_accuracy_pct"),
                                                      (int, float))]
        if vals:
            med = statistics.median(vals)
            spread = f", range [{min(vals):.2f}, {max(vals):.2f}]" if len(vals) > 1 else ""
            print(f"  {arm:8s} n={len(vals)}  median={med:.2f}{spread}")
        else:
            print(f"  {arm:8s} n=0  (no worst-cell number — is eval_envelope set?)")
    print(f"\nWrote {out_dir / 'control_arm_results.json'}")
    print("Both arms spent the same solves per iteration; compare worst-cell "
          "accuracy, and report an honest negative if the treatment does not win.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
