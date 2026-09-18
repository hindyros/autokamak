# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Forced-action meta policies — the active-learning CONTROL ARM.

``docs/active_learning_design.md`` D5 and the system paper's §5.7 promise a
head-to-head: does residual-driven ``enrich_active`` actually beat a blind
``regen_dataset`` append, per expensive solver call? That experiment has
never been run. Every meta workspace on disk uses ``eval_mode:
"seed_shard"`` with ``final_worst_cell_accuracy_pct: null``, and in the one
L0 run that did exercise ``enrich_active`` the refit got WORSE
(7.73e-4 -> 8.15e-4, ``refit_became_best: false``).

The adaptive policies (scripted L0, LLM L1) cannot answer it, because they
*choose* which action to take — so the two arms would differ in their action
sequence as well as their acquisition. These policies remove that freedom:
every data iteration takes the SAME action with the SAME solve count, so the
only thing that varies between arms is HOW the new points are chosen.

Both arms spend iteration 0 on ``extend_search``: without a trained winner
there is no residual signal, so starting on a data action would hand the
random arm an unearned advantage.
"""
from __future__ import annotations

from typing import Literal

ForcedAction = Literal["enrich_active", "regen_dataset"]


def make_forced_meta_policy(
    action: ForcedAction,
    *,
    n_new: int = 200,
    strategy: str = "residual_ucb",
    beta: float = 1.0,
    seed: int = 0,
):
    """Return an ``ActionPicker`` that always takes ``action`` after warm-up.

    Parameters
    ----------
    action
        ``"enrich_active"`` (the treatment: residual-UCB targeted sampling)
        or ``"regen_dataset"`` (the control: blind space-filling append).
    n_new
        New solver calls per data iteration. **Must be identical across
        arms** — the comparison is quality per solve, so an unequal budget
        invalidates it.
    strategy, beta
        Acquisition knobs, used only by the ``enrich_active`` arm.
    """
    if action not in ("enrich_active", "regen_dataset"):
        raise ValueError(
            f"Unknown forced action {action!r}; choose enrich_active or regen_dataset"
        )
    del seed  # deterministic by construction: the action never varies

    from autotokamak.agent.orchestrator.schema import (
        ActionDecision,
        EnrichActivePayload,
        ExtendSearchFocus,
        RegenDatasetOverrides,
    )

    def pick_action(meta_config, state, diagnostics: dict, history: list) -> ActionDecision:
        if state.best_winner_payload is None:
            return ActionDecision(
                action="extend_search",
                extend=ExtendSearchFocus(),
                diagnosis=("CONTROL ARM warm-up: no winner yet, so there is no "
                           "residual signal to target; both arms train first."),
            )
        if action == "enrich_active":
            return ActionDecision(
                action="enrich_active",
                enrich=EnrichActivePayload(
                    n_new=n_new,
                    strategy=strategy,  # type: ignore[arg-type]
                    beta=beta,
                    feasibility_weighting=True,
                    rationale=f"CONTROL ARM (treatment): forced enrich_active, n_new={n_new}",
                ),
                diagnosis=f"CONTROL ARM treatment: adaptive acquisition, {n_new} solves",
            )
        return ActionDecision(
            action="regen_dataset",
            regen=RegenDatasetOverrides(
                overrides={"sampling.n_samples": n_new},
                rationale=f"CONTROL ARM (control): forced blind append, n_new={n_new}",
            ),
            diagnosis=f"CONTROL ARM control: blind space-filling, {n_new} solves",
        )

    return pick_action


__all__ = ["ForcedAction", "make_forced_meta_policy"]
