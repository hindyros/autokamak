# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Tests for the methodology extractor (offline; no LLM, no solver).

The fixtures deliberately reproduce the SHAPES the real corpus produced —
agents log the same three facts under different key names, in different
files, and sometimes only in prose — because every one of those variations
silently zeroed a column during development.
"""
from __future__ import annotations

import json

import pytest

from autotokamak.bench.methodology import (
    build_chain_signature,
    classify,
    extract_meta_methodology,
    extract_methodology,
)


def _write(ws, rel, text):
    p = ws / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _jsonl(records):
    return "\n".join(json.dumps(r) for r in records) + "\n"


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("ensemble_output_variance_mean", "uncertainty_ensemble"),
    ("selected by MC-dropout variance", "uncertainty_ensemble"),
    ("residual-driven UCB on out-of-fold error", "residual_ucb"),
    ("farthest_point distance_to_training", "space_filling"),
])
def test_acquisition_vocabulary_matches_snake_case_and_prose(text, expected):
    # Agents write criteria as identifiers, not sentences; a prose-only
    # pattern set reads "ensemble_output_variance_mean" as no criterion.
    assert expected in classify(text, ["acquisition"])["acquisition"]


def test_classify_unknown_text_yields_nothing():
    assert classify("completely unrelated prose") == {}
    assert classify(None) == {}


# ---------------------------------------------------------------------------
# Per-round decision logic
# ---------------------------------------------------------------------------

def test_rounds_criteria_evidence_and_stop_decision(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    _write(ws, "report.json", json.dumps({
        "sampling_strategy": "LHS then ensemble disagreement",
        "acquisition_log": "logs/acquisition_log.jsonl",
        "adaptive_vs_initial": {"helped": True},
    }))
    records = [{"event": "round_metrics", "round": 0,
                "val_mean_rel_l2": 0.40, "baseline_val_mean_rel_l2": 0.50}]
    for rnd, (val, base) in enumerate([(0.30, 0.50), (0.12, 0.50)], start=1):
        for i in range(3):
            records.append({"event": "acquire_select", "round": rnd, "i": i,
                            "params": {"r0": 0.4}, "sample_id": f"r{rnd}-{i}",
                            "reason": "ensemble_output_variance_mean"})
        records.append({"event": "round_metrics", "round": rnd,
                        "val_mean_rel_l2": val, "baseline_val_mean_rel_l2": base})
    _write(ws, "logs/acquisition_log.jsonl", _jsonl(records))

    m = extract_methodology(ws)
    its = m["iterations"]
    # Round 0 evaluates the initial design; it is not an adaptive decision.
    assert [it["round"] for it in its] == [1, 2]
    assert all(it["n_acquired"] == 3 for it in its)
    assert all(it["criterion_classes"] == ["uncertainty_ensemble"] for it in its)
    assert its[0]["val_over_baseline"] == 0.6
    assert its[0]["decision"] == "continue"
    # 0.12/0.50 = 0.24 <= 0.30 -> the task's own stopping rule was met.
    assert its[1]["met_stop_threshold"] is True
    assert its[1]["decision"] == "stop_threshold_met"
    assert m["decision_logic"]["evidence_grounded_fraction"] == 1.0
    assert m["decision_logic"]["self_claimed_adaptivity_helped"] is True


def test_stop_without_meeting_the_threshold_is_distinguished(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    _write(ws, "acquisition_log.jsonl", _jsonl([
        {"event": "acquire", "round": 1, "params": {}, "reason": "predictive std"},
        {"event": "round_metrics", "round": 1,
         "val_rel_l2": 0.45, "baseline_rel_l2": 0.50},
    ]))
    it = extract_methodology(ws)["iterations"][-1]
    assert it["val_over_baseline"] == 0.9
    assert it["met_stop_threshold"] is False
    assert it["decision"] == "stop_without_threshold"


def test_round_evidence_is_merged_from_a_separate_csv(tmp_path):
    # Decisions in JSONL, per-round errors in a CSV one directory over: the
    # run IS grounded, and reading only the decision log says it is not.
    ws = tmp_path / "workspace"
    ws.mkdir()
    _write(ws, "data/acquisition_log.jsonl", _jsonl([
        {"event": "acquire", "round": 1, "sample_id": "a", "params": {},
         "reason": "minmax(distance_to_training)"},
    ]))
    _write(ws, "data/round_metrics.csv",
           "round,val_baseline_mean,val_model_mean\n1,0.40,0.10\n")
    m = extract_methodology(ws)
    it = m["iterations"][0]
    assert it["baseline_rel_l2"] == 0.40 and it["val_rel_l2"] == 0.10
    assert it["decision"] == "stop_threshold_met"


def test_counts_are_never_mistaken_for_errors(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    _write(ws, "acquisition_log.jsonl", _jsonl([
        {"event": "split_sizes", "round": 1, "n_val": 25, "n_train": 140},
        {"event": "acquire", "round": 1, "params": {}, "reason": "uncertainty"},
    ]))
    it = extract_methodology(ws)["iterations"][0]
    assert it["val_rel_l2"] is None
    assert it["evidence_grounded"] is False
    assert it["decision"] == "stop_unexplained"


def test_criterion_switch_between_rounds_is_recorded(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    _write(ws, "acquisition_log.jsonl", _jsonl([
        {"event": "acquire", "round": 1, "params": {}, "reason": "space_filling coverage"},
        {"event": "acquire", "round": 2, "params": {}, "reason": "residual error targeting"},
    ]))
    logic = extract_methodology(ws)["decision_logic"]
    assert logic["criterion_switched"] is True
    assert set(logic["criterion_classes"]) == {"space_filling", "residual_ucb"}


def test_adaptive_in_name_only_flags_relabelled_random_sampling(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    _write(ws, "acquisition_log.jsonl", _jsonl([
        {"event": "acquire", "round": r, "params": {}, "reason": "random draw"}
        for r in (1, 2)
    ]))
    assert extract_methodology(ws)["decision_logic"]["adaptive_in_name_only"] is True


# ---------------------------------------------------------------------------
# Method chain
# ---------------------------------------------------------------------------

def test_chain_signature_from_code_and_prose(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    _write(ws, "report.json", json.dumps({
        "sampling_strategy": "Latin hypercube design then ensemble disagreement",
    }))
    _write(ws, "train.py",
           "import torch\nfrom sklearn.decomposition import PCA\n"
           "import optuna\nmodel = torch.nn.Linear(5, 32)\n")
    m = extract_methodology(ws)
    chain = m["method_chain"]
    assert chain["model_primary"] == "mlp_torch"
    assert "pca" in chain["representation"]
    assert "optuna" in chain["hpo"]
    assert m["chain_signature"].startswith("lhs -> pca+mlp_torch")
    assert "torch" in m["evidence"]["imports"]


def test_prose_only_claims_are_separated_from_code_evidence(tmp_path):
    # The README claims a GP; nothing in the code is a GP. That gap is the
    # finding, and it must survive into the record rather than being merged
    # away into "the run used a GP".
    ws = tmp_path / "workspace"
    ws.mkdir()
    _write(ws, "README.md", "We fit a Gaussian process surrogate.")
    _write(ws, "train.py", "from sklearn.linear_model import Ridge\n")
    ev = extract_methodology(ws)["evidence"]
    assert "gp" in ev["prose_only_terms"]["model_family"]
    assert "gp" not in (ev["code_terms"].get("model_family") or [])


def test_missing_everything_never_raises(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    m = extract_methodology(ws)
    assert m["iterations"] == []
    assert m["decision_logic"]["n_rounds"] is None
    assert m["chain_signature"]  # a signature of unknowns, not a crash


def test_build_chain_signature_marks_unknown_stages():
    sig = build_chain_signature({"initial_design": [], "model_family": []}, None)
    assert sig.startswith("? -> ?")


# ---------------------------------------------------------------------------
# L0/L1 pipeline cells, in the same vocabulary
# ---------------------------------------------------------------------------

def test_meta_workspace_maps_typed_decisions_onto_the_same_record(tmp_path):
    ws = tmp_path / "L1"
    ws.mkdir()
    _write(ws, "manifest.json", json.dumps({
        "policy": "llm", "n_iterations": 2, "terminated_by": "target_reached",
        "winner_model_name": "gp",
    }))
    _write(ws, "meta_trace.json", json.dumps({"iterations": [
        {"iteration": 0,
         "diagnostics": {"learning_curve": {"interpretation": "still dropping"}},
         "decision": {"action": "extend_search",
                      "extend": {"rationale": "establish a first winner"}}},
        {"iteration": 1,
         "diagnostics": {"pca_spectrum": {"interpretation": "tail energy high"}},
         "decision": {"action": "enrich_active",
                      "enrich": {"strategy": "residual_ucb", "n_new": 500,
                                 "rationale": "target measured weakness"}}},
    ]}))
    m = extract_meta_methodology(ws)
    assert m["decision_logic"]["action_sequence"] == ["extend_search", "enrich_active"]
    assert m["iterations"][1]["criterion_classes"] == ["residual_ucb"]
    assert m["iterations"][1]["n_acquired"] == 500
    # The typed loop hands the picker measurements every iteration.
    assert m["decision_logic"]["evidence_grounded_fraction"] == 1.0
    assert m["method_chain"]["model_primary"] == "gp"
    assert m["decision_logic"]["stop_decision"] == "target_reached"


# ---------------------------------------------------------------------------
# Implemented logic: what the generated CODE computes
# ---------------------------------------------------------------------------

def test_code_logic_reads_the_chooser_not_the_whole_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    # Standardising parameters uses .std() too; only the chooser counts.
    _write(ws, "prep.py", "def standardise(x):\n    return (x - x.mean()) / x.std()\n")
    _write(ws, "acq.py",
           "import numpy as np\n"
           "def acquire_batch(models, pool, k):\n"
           "    preds = np.stack([m.predict(pool) for m in models])\n"
           "    scores = preds.std(axis=0).mean(axis=(1, 2))\n"
           "    return pool[np.argsort(-scores)[:k]]\n")
    code = extract_methodology(ws)["code_logic"]
    assert code["acquisition_implemented"] == ["uncertainty", "uncertainty_ensemble"]
    assert code["selection_rule"] == ["top_k"]
    assert code["model_informed"] is True
    assert code["chooser_functions"][0]["where"].startswith("acq.py:")


def test_model_derived_criteria_require_the_model_to_be_called(tmp_path):
    # Farthest-point selection standardises its inputs with .std(); without
    # this rule that reads as "uncertainty sampling", which it is not.
    ws = tmp_path / "workspace"
    ws.mkdir()
    _write(ws, "acq.py",
           "import numpy as np\n"
           "from scipy.spatial.distance import cdist\n"
           "def propose_batch(train, pool, k):\n"
           "    xs = (pool - train.mean(0)) / train.std(0)\n"
           "    d = cdist(xs, train).min(axis=1)\n"
           "    return pool[np.argsort(-d)[:k]]\n")
    code = extract_methodology(ws)["code_logic"]
    assert code["acquisition_implemented"] == ["space_filling"]
    assert code["model_informed"] is False


def test_stated_criterion_contradicted_by_the_code_is_a_mismatch(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    _write(ws, "report.json", json.dumps({"sampling_strategy": "ensemble disagreement"}))
    _write(ws, "acquisition_log.jsonl", _jsonl([
        {"event": "acquire", "round": 1, "params": {}, "reason": "ensemble disagreement"},
    ]))
    _write(ws, "acq.py",
           "import numpy as np\n"
           "def select_batch(pool, k, rng):\n"
           "    return rng.choice(pool, size=k, replace=False)\n")
    m = extract_methodology(ws)
    assert m["code_logic"]["acquisition_implemented"] == ["random"]
    assert m["stated_vs_implemented"]["verdict"] == "mismatch"
    assert m["stated_vs_implemented"]["only_stated"] == ["uncertainty"]


def test_random_candidate_pool_is_not_read_as_the_criterion(tmp_path):
    # Nearly every implementation draws a random pool and then ranks it.
    ws = tmp_path / "workspace"
    ws.mkdir()
    _write(ws, "acquisition_log.jsonl", _jsonl([
        {"event": "acquire", "round": 1, "params": {}, "reason": "predictive std"},
    ]))
    _write(ws, "acq.py",
           "import numpy as np\n"
           "def acquire(model, k, rng):\n"
           "    pool = rng.uniform(0, 1, size=(500, 5))\n"
           "    psi_mean, psi_std = model.predict(pool, return_std=True)\n"
           "    return pool[np.argsort(-psi_std.mean(axis=1))[:k]]\n")
    m = extract_methodology(ws)
    assert "random" in m["code_logic"]["acquisition_implemented"]
    assert m["stated_vs_implemented"]["verdict"] == "agree"


def test_code_logic_is_unverifiable_rather_than_wrong_when_absent(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    _write(ws, "report.json", json.dumps({"sampling_strategy": "uncertainty sampling"}))
    _write(ws, "train.py", "def train():\n    return 1\n")
    m = extract_methodology(ws)
    assert m["code_logic"]["acquisition_implemented"] == []
    assert m["stated_vs_implemented"]["verdict"] == "unverifiable_from_code"
