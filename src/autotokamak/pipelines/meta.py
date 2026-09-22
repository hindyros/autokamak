# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Meta-loop dispatcher.

Both levels run ``meta_loop.run()`` with ``phase2_mode="structured"``; only
the decision providers differ:

    L0 — scripted meta-action picker + scripted search policy (no LLM)
    L1 — DSPy MetaActionPicker + DSPy SearchRoundPicker

The legacy hybrid (nested Phase-2 via URSA codegen) is still reachable via
``phase2_mode: codegen`` in the meta YAML config; the CLI no longer exposes
it — agent codegen is benchmarked through ``python -m autotokamak.bench``.
"""
from __future__ import annotations

import time
from pathlib import Path

from autotokamak.pipelines._common import (
    REPO_ROOT,
    resolve_output_dir,
    write_manifest,
)

PROMPT_PATH = REPO_ROOT / "src/autotokamak/agent/prompts/surrogate_meta.yaml"


def _effective_config(dataset: str | None, seed: int, out_dir: Path) -> Path:
    """The meta config, with --dataset and --seed actually applied.

    Both flags were accepted by the CLI and then dropped on the floor:
    ``run_meta`` never forwarded ``dataset`` anywhere, and ``meta_loop.run``
    exposes no seed override, so the initial dataset and the train/test
    split seed both came from the committed prompt YAML whatever was asked
    for. A budget-matched baseline run with ``--dataset <450 samples>``
    silently trained on the full 2000-sample committed dataset instead —
    the flag did nothing and said nothing.

    When neither is given the committed prompt is used unchanged, so
    existing behaviour is untouched.
    """
    if dataset is None and seed == 0:
        return PROMPT_PATH
    import yaml

    data = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
    if dataset is not None:
        resolved = Path(dataset)
        if not resolved.is_absolute():
            resolved = (REPO_ROOT / resolved).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"--dataset not found: {resolved}")
        data["initial_dataset_h5"] = str(resolved)
    data["seed"] = int(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "effective_meta_config.yaml"
    dest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"[meta] effective config: {dest} "
          f"(dataset={data.get('initial_dataset_h5')}, seed={data['seed']})")
    return dest


def run_meta(
    *,
    level: str,
    max_iterations: int = 3,
    n_samples: int | None = None,
    enrich_n_new: int | None = None,
    time_budget: int = 600,
    model: str | None = None,
    dataset: str | None = None,
    seed: int = 0,
    target_rmse: float | None = None,
    target_accuracy_pct: float | None = None,
    target_worst_cell_accuracy_pct: float | None = None,
) -> dict:
    """Dispatch the meta-loop at the given access level."""
    from autotokamak.agent.runners.meta_loop import run as meta_run
    from autotokamak.policies import get_meta_policy, get_search_policy

    out_dir = resolve_output_dir("meta", level)
    policy_kind = "scripted" if level == "L0" else "llm"
    config_path = _effective_config(dataset, seed, out_dir)

    pick_action = get_meta_policy(policy_kind, model=model, seed=seed)
    # L0 also replaces the nested extend_search round picker so the whole
    # loop is LLM-free; L1 keeps the DSPy default inside meta_loop.
    phase2_decision_fn = (
        get_search_policy("scripted", seed=seed) if policy_kind == "scripted" else None
    )

    print(f"[meta/{level}] Output: {out_dir}  policy={policy_kind}")

    started = time.time()
    report = meta_run(
        config_path=str(config_path),
        pick_action=pick_action,
        phase2_decision_fn=phase2_decision_fn,
        workspace_override=str(out_dir),
        phase2_mode_override="structured",
        max_iterations_override=max_iterations,
        n_samples_override=n_samples,
        enrich_n_new_override=enrich_n_new,
        phase2_time_budget_override=time_budget,
        model_override=model,
        target_rmse_override=target_rmse,
        target_accuracy_pct_override=target_accuracy_pct,
        target_worst_cell_accuracy_pct_override=target_worst_cell_accuracy_pct,
    )

    elapsed = time.time() - started
    manifest_extra = {
        "elapsed_seconds": round(elapsed, 1),
        "policy": policy_kind,
        "n_iterations": getattr(report, "n_iterations", None),
        "terminated_by": getattr(report, "terminated_by", None),
        "final_rmse": getattr(report, "final_rmse", None),
        "baseline_rmse": getattr(report, "baseline_rmse", None),
        "final_accuracy_pct": getattr(report, "final_accuracy_pct", None),
        "final_worst_cell_accuracy_pct": getattr(report, "final_worst_cell_accuracy_pct", None),
        "winner_model_name": getattr(report, "winner_model_name", None),
        "max_iterations": max_iterations,
        "enrich_n_new": enrich_n_new,
        "time_budget_seconds": time_budget,
        "seed": seed,
    }
    p = write_manifest(out_dir, pipeline="meta", level=level, **manifest_extra)
    print(f"[meta/{level}] Done in {elapsed:.0f}s — manifest: {p}")
    return manifest_extra
