# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Decision providers for the pre-written pipeline (access levels L0 and L1).

The pipeline code is identical across levels; only the decision provider
varies. ``scripted`` (L0) is seeded heuristics with zero LLM calls — the
reproducible golden baseline. ``llm`` (L1) routes the same two decision
points through the DSPy typed pickers.

Two decision points exist:

- **search policy** — one ``RoundDecision`` per Phase-2 AutoML round
  (consumed by ``surrogate.automl_loop.run_automl_loop(decision_fn=...)``).
- **meta policy** — one ``ActionDecision`` per meta-loop iteration
  (consumed by ``agent.runners.meta_loop.run(pick_action=...)``).
"""
from __future__ import annotations

POLICY_KINDS = ("scripted", "llm")
# Meta-policy-only kinds: the active-learning control arm (see forced.py).
FORCED_META_KINDS = ("forced_enrich", "forced_regen")


def get_search_policy(kind: str, *, model: str | None = None, seed: int = 0):
    """Return a ``DecisionFn`` for the Phase-2 AutoML loop."""
    if kind == "scripted":
        from autotokamak.policies.scripted import make_scripted_search_policy

        return make_scripted_search_policy(seed=seed)
    if kind == "llm":
        from autotokamak.policies.llm import make_llm_search_policy

        return make_llm_search_policy(model)
    raise ValueError(f"Unknown policy kind {kind!r}. Choose from {POLICY_KINDS}")


def get_meta_policy(kind: str, *, model: str | None = None, seed: int = 0,
                    forced_n_new: int = 200):
    """Return an ``ActionPicker`` for the meta-loop."""
    if kind == "scripted":
        from autotokamak.policies.scripted import make_scripted_meta_policy

        return make_scripted_meta_policy(seed=seed)
    if kind == "llm":
        from autotokamak.policies.llm import make_llm_meta_policy

        return make_llm_meta_policy()
    if kind in FORCED_META_KINDS:
        from autotokamak.policies.forced import make_forced_meta_policy

        return make_forced_meta_policy(
            "enrich_active" if kind == "forced_enrich" else "regen_dataset",
            n_new=forced_n_new,
            seed=seed,
        )
    raise ValueError(
        f"Unknown policy kind {kind!r}. Choose from "
        f"{POLICY_KINDS + FORCED_META_KINDS}"
    )
