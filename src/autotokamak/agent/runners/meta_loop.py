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
    """Default action-picker. Calls the LLM with structured output.

    The LLM is shown the current diagnostics and prior history; it returns a
    validated ``ActionDecision``. Errors propagate so the runner can record
    them in the trace.
    """
    from langchain.chat_models import init_chat_model

    llm = init_chat_model(model=meta_config.model)
    # Use function_calling instead of OpenAI's strict json_schema mode because
    # ActionDecision contains a Dict[str, Any] overrides field (free-form dotted
    # keys) that strict mode rejects for missing additionalProperties=false.
    structured = llm.with_structured_output(ActionDecision, method="function_calling")

    iterations_remaining = meta_config.max_iterations - len(history)
    prompt = _build_action_prompt(state, diagnostics, history, iterations_remaining)
    return structured.invoke(prompt)


def _build_action_prompt(
    state: MetaState,
    diagnostics: dict,
    history: list[MetaIterationRecord],
    iterations_remaining: int,
) -> str:
    """Compose the per-iteration prompt for the action picker."""
    summary = {
        "iteration": len(history),
        "iterations_remaining": iterations_remaining,
        "current_dataset": str(state.current_dataset_h5),
        "best_rmse_so_far": (
            None if state.best_rmse == float("inf") else state.best_rmse
        ),
        "rmse_history": list(state.rmse_history),
        "actions_taken": list(state.actions_taken),
        "diagnostics": diagnostics,
        "prior_decisions": [
            {
                "iteration": r.iteration,
                "action": r.decision.action,
                "diagnosis": r.decision.diagnosis,
                "rmse_after": r.rmse_after,
            }
            for r in history
        ],
    }
    return (
        "You are the META-AGENT orchestrating a Grad-Shafranov surrogate-model "
        "improvement loop. Each iteration you must choose ONE action:\n"
        "  - regen_dataset(overrides): regenerate the Phase-1 dataset with "
        "    overrides applied to the base sweep config. Use this when "
        "    diagnostics say the surrogate is sample-bottlenecked or has high "
        "    cross-seed variance (more data needed).\n"
        "  - extend_search(focus): run another Phase-2 search emphasizing "
        "    specific models or widening certain hyperparameter ranges. Use "
        "    this when edge_hit_summary shows persistent edge hits OR when "
        "    learning_curve has plateaued but RMSE is still above baseline.\n"
        "  - terminate(reason): stop the loop. Use this when the surrogate is "
        "    clearly good enough or further actions would not help.\n\n"
        f"State and diagnostics:\n{json.dumps(summary, indent=2, default=str)}\n\n"
        "Return an ActionDecision JSON. Always include a one-sentence diagnosis "
        "explaining what you think the bottleneck is."
    )


def measure_test_rmse(state: MetaState) -> Optional[float]:
    """Re-evaluate the current best winner on the CURRENT dataset's test split.

    Returns None if no winner is available yet. The RMSE is computed against
    the dataset state.current_dataset_h5, which may have changed since the
    winner was trained — so this measures generalization to the live data.
    """
    if state.best_winner_payload is None:
        return None
    try:
        bundle = load_dataset(state.current_dataset_h5)
        splits = kfold(bundle, k=4, test_frac=2 / bundle.n_samples, seed=state.seed)
        from autotokamak.surrogate.automl import predict_with_winner

        pred = predict_with_winner(state.best_winner_payload, bundle.inputs[splits.test_idx])
        return float(psi_rmse(bundle.psi[splits.test_idx], pred))
    except Exception:  # noqa: BLE001
        return None


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
) -> MetaReport:
    """Run the meta-loop. Returns the final ``MetaReport``.

    ``model_override`` and ``max_iterations_override``, if set, win over the
    values in the meta YAML — convenient for cheap test runs.
    """
    meta_config = MetaConfig.from_yaml(config_path)
    if model_override:
        meta_config = meta_config.model_copy(update={"model": model_override})
    if max_iterations_override is not None:
        meta_config = meta_config.model_copy(update={"max_iterations": int(max_iterations_override)})

    workspace_path = resolve_workspace(meta_config.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    (workspace_path / "iterations").mkdir(exist_ok=True)
    (workspace_path / "datasets").mkdir(exist_ok=True)
    (workspace_path / "surrogate_runs").mkdir(exist_ok=True)

    # Resolve initial dataset path (relative to repo root if not absolute).
    initial_dataset = Path(meta_config.initial_dataset_h5)
    if not initial_dataset.is_absolute():
        initial_dataset = (REPO_ROOT / initial_dataset).resolve()

    base_sweep = None
    if meta_config.base_sweep_config:
        base_path = Path(meta_config.base_sweep_config)
        if not base_path.is_absolute():
            base_path = (REPO_ROOT / base_path).resolve()
        base_sweep = SweepConfig.from_yaml(base_path)

    state = MetaState(
        workspace=workspace_path,
        current_dataset_h5=initial_dataset,
        base_sweep_config=base_sweep,
        phase2_prompt=(REPO_ROOT / meta_config.phase2_prompt),
        seed=meta_config.seed,
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

    # Initial baseline RMSE so we have something to report even at iteration 0.
    bundle = load_dataset(state.current_dataset_h5)
    splits0 = kfold(bundle, k=4, test_frac=2 / bundle.n_samples, seed=state.seed)
    baseline_rmse = float(
        sum(
            baseline_mean_predictor_rmse(bundle.psi[tr], bundle.psi[va])
            for _, tr, va in splits0.iter_folds()
        ) / len(splits0.folds)
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
            decision = pick_action(meta_config, state, diag, history)
            (iter_dir / "action.json").write_text(decision.model_dump_json(indent=2))
            print(f"  action: {decision.action}; diagnosis: {decision.diagnosis}")

            record = MetaIterationRecord(
                iteration=i,
                started_utc=_now(),
                diagnostics=diag,
                decision=decision,
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

        # Final refit on the latest dataset state, if we have any winner.
        winner_path = workspace_path / "winner.pkl"
        if state.best_winner_payload is not None and state.best_winner_path is not None:
            import shutil

            shutil.copy(state.best_winner_path, winner_path)

        final_rmse = state.best_rmse if state.best_rmse != float("inf") else baseline_rmse
        actions_taken_typed = []
        for a in state.actions_taken:
            if a in {"regen_dataset", "extend_search", "terminate"}:
                actions_taken_typed.append(a)
        report = MetaReport(
            n_iterations=len(history),
            terminated_by=terminated_by,
            final_rmse=float(final_rmse),
            baseline_rmse=float(baseline_rmse),
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
        print(f"  final RMSE: {final_rmse:.4f}; baseline: {baseline_rmse:.4f}")
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
    args = parser.parse_args()

    run(
        config_path=args.config,
        trace_enabled=not args.no_trace,
        experiments_dir=Path(args.experiments_dir) if args.experiments_dir else None,
        model_override=args.model,
        max_iterations_override=args.max_iterations,
    )


if __name__ == "__main__":
    main()
