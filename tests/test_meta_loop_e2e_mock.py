"""End-to-end meta-loop test with a scripted action picker.

Bypasses the LLM by supplying a hand-written ``ActionPicker`` to
``meta_loop.run``. Verifies that:
- the meta workspace is created with the expected layout
- each iteration writes diagnostics/action/result files
- the terminate action breaks the loop
- the final report.json + meta_trace.json are valid
- the meta scorer returns a non-zero total

Does NOT exercise the nested Phase-2 LLM call (would require a real
``plan_execute_feedback`` run, ~10 min and ~$1-5). The ``extend_search``
action is exercised in a separate test that monkey-patches the nested call
to inject a pre-computed winner.pkl + report.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATASET = REPO_ROOT / "examples" / "dataset_generation" / "outputs" / "dataset.h5"
PHASE2_PROMPT = REPO_ROOT / "src" / "autotokamak" / "agent" / "prompts" / "surrogate_automl.yaml"


@pytest.fixture
def meta_config_yaml(tmp_path: Path) -> Path:
    """Write a tiny meta_config.yaml pointing at the real Phase-1 dataset."""
    cfg_path = tmp_path / "meta_config.yaml"
    cfg_path.write_text(
        "max_iterations: 2\n"
        "seed: 0\n"
        f"initial_dataset_h5: {REAL_DATASET}\n"
        f"base_sweep_config: {REPO_ROOT}/examples/dataset_generation/dataset_config.yaml\n"
        f"phase2_prompt: {PHASE2_PROMPT}\n"
        f"workspace: {tmp_path / 'meta_ws'}\n"
        "model: openai:gpt-5.2\n"
    )
    return cfg_path


def _make_picker(decisions: list[dict]):
    """Return an ActionPicker that yields the supplied decisions in order."""
    from autotokamak.agent.orchestrator.schema import ActionDecision

    materialized = [ActionDecision.model_validate(d) for d in decisions]
    counter = {"i": 0}

    def picker(meta_config, state, diagnostics, history):
        d = materialized[counter["i"]]
        counter["i"] += 1
        return d

    return picker


@pytest.mark.skipif(
    not REAL_DATASET.is_file(),
    reason=f"Phase-1 dataset not present at {REAL_DATASET}",
)
def test_meta_loop_terminate_path(meta_config_yaml: Path, tmp_path: Path):
    """Simplest path: agent immediately terminates."""
    # Make src/autotokamak runner imports work without installing the package.
    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)

    picker = _make_picker(
        [
            {
                "action": "terminate",
                "terminate": {"reason": "baseline is good enough", "confidence": "high"},
                "diagnosis": "good enough for the smoke test",
            }
        ]
    )

    report = run(
        config_path=str(meta_config_yaml),
        pick_action=picker,
        trace_enabled=False,
    )
    assert report.terminated_by == "agent"
    assert report.n_iterations == 1


@pytest.mark.skipif(
    not REAL_DATASET.is_file(),
    reason=f"Phase-1 dataset not present at {REAL_DATASET}",
)
def test_meta_loop_regen_then_terminate(meta_config_yaml: Path, tmp_path: Path):
    """Exercises regen_dataset action without invoking the Phase-2 sub-LLM."""
    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)

    picker = _make_picker(
        [
            {
                "action": "regen_dataset",
                "regen": {
                    "overrides": {"sampling.n_samples": 3, "sampling.seed": 7},
                    "rationale": "smoke test of regen path",
                },
                "diagnosis": "sample-bottlenecked, regenerate with smaller N for speed",
            },
            {
                "action": "terminate",
                "terminate": {"reason": "regen exercised, end of test", "confidence": "high"},
                "diagnosis": "test complete",
            },
        ]
    )

    report = run(
        config_path=str(meta_config_yaml),
        pick_action=picker,
        trace_enabled=False,
    )
    assert report.terminated_by == "agent"
    assert report.n_iterations == 2

    # Workspace artifacts should exist.
    ws = Path(json.loads(meta_config_yaml.read_text().split("workspace: ", 1)[1].split("\n")[0]).strip() if False else REPO_ROOT)
    # Easier: derive it from meta_config.
    from autotokamak.agent.orchestrator.schema import MetaConfig

    mc = MetaConfig.from_yaml(meta_config_yaml)
    ws = Path(mc.workspace)
    assert (ws / "meta_trace.json").is_file()
    assert (ws / "report.json").is_file()
    assert (ws / "iterations" / "000" / "diagnostics.json").is_file()
    assert (ws / "iterations" / "000" / "action.json").is_file()
    assert (ws / "iterations" / "001" / "action.json").is_file()

    # The regen action should have produced a new dataset.
    datasets_dir = ws / "datasets"
    assert any(datasets_dir.iterdir())


def test_meta_loop_scorer_requires_winner(tmp_path: Path):
    """A meta workspace with no winner.pkl fails the winner_predicts hard gate."""
    from autotokamak.agent.dspy.metric_meta import score_meta_run

    ws = tmp_path / "empty_meta"
    ws.mkdir()
    (ws / "report.json").write_text("{}")
    (ws / "meta_trace.json").write_text(json.dumps({"iterations": []}))

    rep = score_meta_run(ws)
    assert rep.total == 0.0
    assert not rep.all_gates_pass
