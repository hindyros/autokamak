"""Meta-loop runner — the autonomous outer loop above Phase-2.

Per iteration:
  1. Compute diagnostics (deterministic).
  2. Ask the LLM for an ``ActionDecision`` (structured output).
  3. Dispatch the chosen action.
  4. Measure and record into the meta-trace.

The action-picker LLM call is factored into ``pick_action`` so tests can
inject a hand-written decision sequence without standing up langchain.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from autotokamak.agent.orchestrator.actions import MetaState, dispatch
from autotokamak.agent.orchestrator.schema import (
    ActionDecision,
    MetaConfig,
    MetaIterationRecord,
    MetaReport,
)
from autotokamak.data.schema import SweepConfig
from autotokamak.eval.data import kfold, load_dataset
from autotokamak.eval.metrics import baseline_mean_predictor_rmse, psi_rmse

from agent.runners.config import REPO_ROOT, resolve_workspace
from agent.runners.trace import RunTrace

load_dotenv(REPO_ROOT / ".env")


DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "experiments"
ActionPicker = Callable[[MetaConfig, MetaState, dict, list[MetaIterationRecord]], ActionDecision]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def pick_action_via_llm(
    meta_config: MetaConfig,
    state: MetaState,
    diagnostics: dict,
    history: list[MetaIterationRecord],
) -> ActionDecision:
    """Default action-picker. Delegates to the DSPy MetaActionPickerModule.

    Behavior:
      - Loads optimized prompt state from
        ``agent/dspy/optimized/meta_picker.json`` if present (post-GEPA).
      - Falls back to the in-code baseline (signature docstring at
        ``agent/dspy/signatures.py``) when no optimized state exists.
      - Honors ``state.use_baseline_picker`` (set via the runner's
        ``--use-baseline`` flag) to force the in-code baseline for A/B
        comparison even when an optimized JSON exists.

    The LM is configured via ``dspy.configure(lm=...)`` lazily on first call.
    """
    import dspy

    from autotokamak.agent.dspy.module import (
        DEFAULT_OPTIMIZED_PATH,
        MetaActionPickerModule,
        load_module,
    )

    # Configure DSPy's LM once per process. dspy.LM uses litellm-style strings
    # ("openai/gpt-5-mini"); convert from our "openai:gpt-5-mini" convention.
    settings_lm = getattr(dspy.settings, "lm", None)
    desired_lm_string = meta_config.model.replace(":", "/", 1)
    if settings_lm is None or getattr(settings_lm, "model", None) != desired_lm_string:
        dspy.configure(lm=dspy.LM(desired_lm_string))

    use_baseline = bool(getattr(state, "use_baseline_picker", False))
    if use_baseline or not DEFAULT_OPTIMIZED_PATH.is_file():
        module = MetaActionPickerModule()
    else:
        module = load_module(DEFAULT_OPTIMIZED_PATH)

    # Single source of truth for the LM inputs — the runner records the same
    # strings into the trace, so GEPA trains on byte-identical inputs.
    from autotokamak.agent.dspy.picker_inputs import picker_inputs_from_runtime

    inputs = picker_inputs_from_runtime(meta_config, state, diagnostics, history)
    return module.predict_action_decision(**inputs)


def measure_test_rmse(state: MetaState) -> Optional[float]:
    """Evaluate the current best winner on the FROZEN test shard.

    Returns None if no winner or no shard is available. The shard is carved
    once from the initial dataset and never grows or reshuffles, so RMSEs
    are comparable across iterations regardless of how the train pool has
    changed — and no winner-training sample can leak into it.

    Delegates to ``actions._frozen_shard_rmse`` — the loop's single most
    important measurement must have exactly one implementation.
    """
    from autotokamak.agent.orchestrator.actions import _frozen_shard_rmse

    if state.best_winner_payload is None:
        return None
    return _frozen_shard_rmse(state.best_winner_payload, state)


def _initial_diagnostics(state: MetaState) -> dict:
    """Cheap diagnostics that don't require an existing winner."""
    from autotokamak.eval import diagnostics as diag_mod
    from autotokamak.surrogate.zoo import make_poly_ridge

    bundle = load_dataset(state.current_dataset_h5)
    factory = lambda: make_poly_ridge(alpha=0.1, degree=2)
    return diag_mod.run_all(bundle, model_factory=factory)


def _diagnostics_with_winner(state: MetaState) -> dict:
    """Full diagnostics including residual structure (requires a winner)."""
    from autotokamak.eval import diagnostics as diag_mod
    from autotokamak.surrogate.zoo import make_poly_ridge

    bundle = load_dataset(state.current_dataset_h5)
    factory = lambda: make_poly_ridge(alpha=0.1, degree=2)
    splits = None
    if state.best_winner_payload is not None:
        splits = kfold(bundle, k=4, test_frac=2 / bundle.n_samples, seed=state.seed)
    return diag_mod.run_all(
        bundle,
        model_factory=factory,
        winner_payload=state.best_winner_payload,
        splits=splits,
    )


def run(
    config_path: str,
    *,
    pick_action: ActionPicker = pick_action_via_llm,
    trace_enabled: bool = True,
    experiments_dir: Optional[Path] = None,
    model_override: Optional[str] = None,
    max_iterations_override: Optional[int] = None,
    n_samples_override: Optional[int] = None,
    phase2_time_budget_override: Optional[int] = None,
    use_baseline_picker: bool = False,
    workspace_override: Optional[str] = None,
    phase2_mode_override: Optional[str] = None,
    target_rmse_override: Optional[float] = None,
    target_rmse_ratio_override: Optional[float] = None,
) -> MetaReport:
    """Run the meta-loop. Returns the final ``MetaReport``.

    ``model_override`` and ``max_iterations_override``, if set, win over the
    values in the meta YAML — convenient for cheap test runs.

    ``phase2_mode_override`` selects the Phase-2 execution strategy:
      "structured" (default) — Optuna + DSPy library (fast, no code-gen)
      "codegen"             — URSA PlanningAgent + ExecutionAgent writes the runner

    ``use_baseline_picker=True`` forces the in-code baseline DSPy module
    (ignoring any saved optimized prompt). Used for A/B comparison after
    GEPA optimization.
    """
    meta_config = MetaConfig.from_yaml(config_path)
    if model_override:
        meta_config = meta_config.model_copy(update={"model": model_override})
    if max_iterations_override is not None:
        meta_config = meta_config.model_copy(update={"max_iterations": int(max_iterations_override)})
    if phase2_mode_override is not None:
        if phase2_mode_override not in ("structured", "codegen"):
            raise ValueError(
                f"phase2_mode_override must be 'structured' or 'codegen', got {phase2_mode_override!r}"
            )
        meta_config = meta_config.model_copy(update={"phase2_mode": phase2_mode_override})
    if target_rmse_override is not None:
        meta_config = meta_config.model_copy(update={"target_rmse": float(target_rmse_override)})
    if target_rmse_ratio_override is not None:
        meta_config = meta_config.model_copy(
            update={"target_rmse_ratio": float(target_rmse_ratio_override)}
        )

    workspace_path = resolve_workspace(workspace_override or meta_config.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    (workspace_path / "iterations").mkdir(exist_ok=True)
    (workspace_path / "datasets").mkdir(exist_ok=True)
    (workspace_path / "surrogate_runs").mkdir(exist_ok=True)

    # Resolve initial dataset path (relative to repo root if not absolute).
    initial_dataset = Path(meta_config.initial_dataset_h5)
    if not initial_dataset.is_absolute():
        initial_dataset = (REPO_ROOT / initial_dataset).resolve()

    # Freeze the held-out test shard BEFORE anything else touches the data.
    # Every RMSE the loop reports (baseline, per-iteration, final) is measured
    # on this fixed shard; the remainder becomes the growing train pool. A
    # dataset too small to split fails fast — a meta run without a
    # trustworthy shard produces incomparable metrics.
    from autotokamak.data.h5io import split_h5

    datasets_dir = workspace_path / "datasets"
    train_pool = datasets_dir / "train_pool.h5"
    test_shard = datasets_dir / "test_shard.h5"
    split_info = split_h5(
        initial_dataset,
        train_path=train_pool,
        test_path=test_shard,
        test_frac=meta_config.holdout_test_frac,
        min_test=meta_config.holdout_min_test,
        seed=meta_config.seed,
    )
    (datasets_dir / "split_info.json").write_text(
        json.dumps(split_info, indent=2, default=str)
    )
    print(
        f"Frozen test shard: {test_shard} ({split_info['n_test']} samples); "
        f"train pool: {split_info['n_train_success']} successful samples"
    )

    base_sweep = None
    if meta_config.base_sweep_config:
        base_path = Path(meta_config.base_sweep_config)
        if not base_path.is_absolute():
            base_path = (REPO_ROOT / base_path).resolve()
        base_sweep = SweepConfig.from_yaml(base_path)
        if n_samples_override is not None:
            bumped = base_sweep.sampling.model_copy(
                update={"n_samples": int(n_samples_override)}
            )
            base_sweep = base_sweep.model_copy(update={"sampling": bumped})

    state = MetaState(
        workspace=workspace_path,
        current_dataset_h5=train_pool,
        base_sweep_config=base_sweep,
        phase2_prompt=(REPO_ROOT / meta_config.phase2_prompt),
        seed=meta_config.seed,
        test_shard_h5=test_shard,
        phase2_mode=meta_config.phase2_mode,
        phase2_model=meta_config.model,
        phase2_max_rounds=meta_config.phase2_max_rounds,
        phase2_time_budget_seconds=(
            int(phase2_time_budget_override)
            if phase2_time_budget_override is not None
            else None
        ),
        use_baseline_picker=bool(use_baseline_picker),
    )

    trace = None
    if trace_enabled:
        try:
            target_dir = experiments_dir or DEFAULT_EXPERIMENTS_DIR
            trace = RunTrace.open(
                experiments_dir=target_dir,
                prompt_path=(REPO_ROOT / config_path
                             if not str(config_path).startswith("/") else Path(config_path)),
                model=meta_config.model,
                feedback_rounds=meta_config.max_iterations,
                workspace=str(workspace_path),
            )
            print(f"Meta-trace: {trace._path}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: failed to open meta-trace ({exc})", file=sys.stderr)
            trace = None

    # Baseline: mean-predictor of the train pool evaluated on the frozen
    # shard — the same shard every winner is measured on, so
    # final_rmse / baseline_rmse is apples-to-apples.
    train_bundle = load_dataset(state.current_dataset_h5)
    shard_bundle = load_dataset(state.test_shard_h5)
    baseline_rmse = float(
        baseline_mean_predictor_rmse(train_bundle.psi, shard_bundle.psi)
    )

    # Resolve the early-stopping quality bar (absolute wins the min if both
    # forms are set). The loop stops as soon as the frozen-shard RMSE meets
    # it — no point spending further iterations once the model is good enough.
    target_candidates = []
    if meta_config.target_rmse is not None:
        target_candidates.append(float(meta_config.target_rmse))
    if meta_config.target_rmse_ratio is not None:
        target_candidates.append(float(meta_config.target_rmse_ratio) * baseline_rmse)
    target_rmse_abs = min(target_candidates) if target_candidates else None
    state.target_rmse_abs = target_rmse_abs
    if target_rmse_abs is not None:
        print(
            f"Early-stop target: shard RMSE <= {target_rmse_abs:.6g} "
            f"(baseline {baseline_rmse:.6g})"
        )

    history: list[MetaIterationRecord] = []
    terminated_by: str = "iterations_cap"

    try:
        for i in range(meta_config.max_iterations):
            iter_dir = workspace_path / "iterations" / f"{i:03d}"
            iter_dir.mkdir(exist_ok=True)

            diag = (
                _diagnostics_with_winner(state)
                if state.best_winner_payload is not None
                else _initial_diagnostics(state)
            )
            (iter_dir / "diagnostics.json").write_text(
                json.dumps(diag, indent=2, default=str)
            )

            print(f"\n=== META iteration {i} (of {meta_config.max_iterations}) ===")
            # Record the EXACT LM inputs for this decision (regardless of which
            # picker is plugged in) so GEPA later trains on byte-identical
            # inputs — see agent/dspy/picker_inputs.py (train/serve skew fix).
            from autotokamak.agent.dspy.picker_inputs import picker_inputs_from_runtime

            picker_inputs = picker_inputs_from_runtime(meta_config, state, diag, history)
            decision = pick_action(meta_config, state, diag, history)
            (iter_dir / "action.json").write_text(decision.model_dump_json(indent=2))
            print(f"  action: {decision.action}; diagnosis: {decision.diagnosis}")

            record = MetaIterationRecord(
                iteration=i,
                started_utc=_now(),
                diagnostics=diag,
                decision=decision,
                picker_inputs=picker_inputs,
            )

            if decision.action == "terminate":
                record.finished_utc = _now()
                record.result = {"kind": "terminate"}
                history.append(record)
                terminated_by = "agent"
                break

            try:
                result = dispatch(decision, state)
            except Exception as exc:  # noqa: BLE001
                result = {"kind": "error", "error": f"{type(exc).__name__}: {exc}"}
                print(f"  action dispatch failed: {result['error']}", file=sys.stderr)

            (iter_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
            record.result = result

            state.actions_taken.append(decision.action)
            rmse_after = measure_test_rmse(state)
            record.rmse_after = rmse_after
            if rmse_after is not None:
                state.rmse_history.append(rmse_after)

            record.finished_utc = _now()
            history.append(record)

            if (
                target_rmse_abs is not None
                and rmse_after is not None
                and rmse_after <= target_rmse_abs
            ):
                terminated_by = "target_reached"
                print(
                    f"  target reached: shard RMSE {rmse_after:.6g} <= "
                    f"{target_rmse_abs:.6g}; stopping."
                )
                break

        # Final refit on the latest dataset state, if we have any winner.
        winner_path = workspace_path / "winner.pkl"
        if state.best_winner_payload is not None and state.best_winner_path is not None:
            import shutil

            shutil.copy(state.best_winner_path, winner_path)

        # None (not baseline) when no winner was ever produced — falling back
        # to baseline_rmse here made a winnerless run read as "matched
        # baseline" in the report.
        final_rmse = state.best_rmse if state.best_rmse != float("inf") else None
        actions_taken_typed = []
        for a in state.actions_taken:
            if a in {"regen_dataset", "extend_search", "terminate"}:
                actions_taken_typed.append(a)
        report = MetaReport(
            n_iterations=len(history),
            terminated_by=terminated_by,
            target_rmse=target_rmse_abs,
            final_rmse=final_rmse,
            baseline_rmse=float(baseline_rmse),
            test_shard_path=str(test_shard),
            n_test_samples=int(split_info["n_test"]),
            n_train_pool_samples=int(split_info["n_train_success"]),
            initial_rmse=state.rmse_history[0] if state.rmse_history else None,
            winner_model_name=(
                state.best_surrogate_report.get("winner_model_name", "none")
                if state.best_surrogate_report else "none"
            ),
            winner_hyperparams=(
                state.best_surrogate_report.get("winner_hyperparams", {})
                if state.best_surrogate_report else {}
            ),
            rmse_history=list(state.rmse_history),
            actions_taken=actions_taken_typed,
        )
        report_path = workspace_path / "report.json"
        report_path.write_text(report.model_dump_json(indent=2))

        # Write a meta_trace.json with the full per-iteration log alongside the
        # RunTrace.json (which only records the outer skeleton).
        (workspace_path / "meta_trace.json").write_text(
            json.dumps(
                {
                    "iterations": [r.model_dump(mode="json") for r in history],
                    "report": report.model_dump(mode="json"),
                },
                indent=2,
                default=str,
            )
        )

        if trace:
            trace.record_artifacts(
                workspace_path,
                expected_artifacts=["winner.pkl", "report.json", "meta_trace.json"],
            )
            try:
                from autotokamak.agent.dspy.metric_meta import score_meta_run

                score = score_meta_run(workspace_path)
                trace.record_score(score)
                print(f"\nMeta score: {score.total:.3f}")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: meta scoring failed: {exc}", file=sys.stderr)
            trace.mark_completed()

        print(f"\n=== META FINAL ===")
        print(f"  iterations: {len(history)}; terminated_by: {terminated_by}")
        final_str = f"{final_rmse:.4f}" if final_rmse is not None else "n/a (no winner)"
        print(f"  final RMSE: {final_str}; baseline: {baseline_rmse:.4f} "
              f"(both on frozen shard, n={split_info['n_test']})")
        return report

    except KeyboardInterrupt:
        if trace:
            try:
                trace.record_artifacts(workspace_path)
            except Exception:  # noqa: BLE001
                pass
            trace.mark_interrupted()
        raise
    except Exception as exc:
        if trace:
            try:
                trace.record_artifacts(workspace_path)
            except Exception:  # noqa: BLE001
                pass
            trace.mark_errored(exc)
        raise


def main():
    parser = argparse.ArgumentParser(description="Run the autotokamak meta-agent loop.")
    parser.add_argument("--config", required=True, help="Path to surrogate_meta.yaml")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--experiments-dir", default=None)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max_iterations from the config (1 = cheapest real test).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the LLM model (e.g. 'openai:gpt-5-mini' for cheap testing).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override sampling.n_samples in the loaded base_sweep_config "
             "(effective on the next regen_dataset action).",
    )
    parser.add_argument(
        "--time-budget-seconds",
        type=int,
        default=None,
        help="Injected as a HARD BUDGET directive into every extend_search "
             "overlay prompt so the Phase-2 agent writes it into "
             "surrogate_config.yaml.",
    )
    parser.add_argument(
        "--use-baseline",
        action="store_true",
        help="Force the in-code baseline action-picker prompt (ignore any optimized JSON). "
             "Used for A/B comparison after GEPA optimization.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override the workspace from the config. Required for trace "
             "collection: each run needs its OWN workspace or successive runs "
             "clobber each other's meta_trace.json.",
    )
    parser.add_argument(
        "--target-rmse",
        type=float,
        default=None,
        help="Early-stop when the frozen-shard RMSE reaches this ABSOLUTE "
             "value (dataset psi units).",
    )
    parser.add_argument(
        "--target-rmse-ratio",
        type=float,
        default=None,
        help="Early-stop when shard RMSE <= ratio * baseline RMSE "
             "(scale-free; e.g. 0.3).",
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "ursa"),
        default=None,
        help="Phase-2 execution mode: 'fast' (Optuna+DSPy library, default) or "
             "'ursa' (hybrid: URSA code-gen for nested Phase-2 searches). "
             "Equivalent to --phase2-mode structured|codegen.",
    )
    args = parser.parse_args()

    phase2_mode = None
    if args.mode == "fast":
        phase2_mode = "structured"
    elif args.mode == "ursa":
        phase2_mode = "codegen"

    run(
        config_path=args.config,
        trace_enabled=not args.no_trace,
        experiments_dir=Path(args.experiments_dir) if args.experiments_dir else None,
        model_override=args.model,
        max_iterations_override=args.max_iterations,
        n_samples_override=args.n_samples,
        phase2_time_budget_override=args.time_budget_seconds,
        use_baseline_picker=args.use_baseline,
        workspace_override=args.workspace,
        phase2_mode_override=phase2_mode,
        target_rmse_override=args.target_rmse,
        target_rmse_ratio_override=args.target_rmse_ratio,
    )


if __name__ == "__main__":
    main()
