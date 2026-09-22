# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""The glossary must cover every code the analysis can print.

A report cell reading ``claimed_only`` or ``per_worker_process`` is useless to
a reader who cannot look it up, so these tests are drift guards: add a value
to a detector or a term to a pattern table without defining it and they fail.
"""
from __future__ import annotations

import inspect
import re

import pytest

from autotokamak.bench import methodology as M
from autotokamak.bench import solution_shape as S
from autotokamak.bench.glossary import (
    all_terms,
    define,
    describe_value,
    dimension_help,
    split_value,
)

# Values whose text carries a run-specific count; defined by pattern instead.
DYNAMIC = re.compile(r"\{|_modules$|_runnable_scripts$")


def test_every_value_a_detector_can_emit_is_defined():
    src = inspect.getsource(S)
    emitted = set(re.findall(r'_dim\(\s*"([a-z0-9_+]+)"', src))
    missing = sorted(
        tok
        for value in emitted
        for tok in value.split("+")
        if not DYNAMIC.search(tok) and tok not in S.VALUE_GLOSSARY
    )
    assert not missing, f"undefined solution-shape values: {missing}"


def test_every_dimension_has_a_question_and_a_note():
    for dim in S.DIMENSIONS:
        assert S.DIMENSION_QUESTIONS.get(dim), f"{dim} has no question"
        assert S.DIMENSION_NOTES.get(dim), f"{dim} has no note"
        assert dimension_help(dim)


@pytest.mark.parametrize("table", [
    "ACQUISITION_PATTERNS", "INITIAL_DESIGN_PATTERNS", "REPRESENTATION_PATTERNS",
    "MODEL_PATTERNS", "HPO_PATTERNS", "ENSEMBLE_PATTERNS", "STOP_RULE_PATTERNS",
    "CODE_LOGIC_PATTERNS", "SELECTION_PATTERNS",
])
def test_every_canonical_term_is_defined(table):
    missing = sorted(k for k in getattr(M, table) if k not in M.TERM_GLOSSARY)
    assert not missing, f"undefined terms in {table}: {missing}"


def test_per_round_decisions_and_verdicts_are_defined():
    for token in ("continue", "stop_threshold_met", "stop_without_threshold",
                  "stop_unexplained", "agree", "partial", "mismatch",
                  "unverifiable_from_code", "undocumented"):
        assert define(token), f"{token} has no definition"


def test_composite_values_resolve_part_by_part():
    assert split_value("npz_per_solve+pickle+index") == [
        "npz_per_solve", "pickle", "index"]
    text = describe_value("npz_per_solve+pickle+index")
    assert "npz_per_solve —" in text and "index —" in text


def test_counted_values_resolve_by_pattern():
    assert "11 source files" in (define("11_modules") or "")
    assert "3 files are runnable" in (define("3_runnable_scripts") or "")


def test_unknown_token_returns_none_rather_than_inventing_a_meaning():
    assert define("not_a_real_code") is None
    assert "(no definition recorded)" in describe_value("not_a_real_code")


def test_all_terms_groups_are_non_empty():
    groups = all_terms()
    assert len(groups) == 2
    assert all(len(v) > 10 for v in groups.values())
