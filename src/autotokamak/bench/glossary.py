# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""One lookup for every code this benchmark prints.

The reports are dense with canonical tokens — ``lcfs_polygon``,
``residual_ucb``, ``stop_without_threshold``, ``npz_per_solve+index``. A code
a reader has to guess at is not a measurement, so every one of them is
defined next to the detector that emits it, and this module is the single
place that resolves a token (or a composite value) to those definitions for
rendering.

Composite and counted values are handled here too: ``a+b`` is two tokens, and
``11_modules`` is a shape whose meaning is the same whatever the number.
"""
from __future__ import annotations

import re

from autotokamak.bench.methodology import TERM_GLOSSARY
from autotokamak.bench.solution_shape import (
    DIMENSION_NOTES,
    DIMENSION_QUESTIONS,
    VALUE_GLOSSARY,
)

# Patterns for values that carry a count. The number varies per run; what the
# value MEANS does not.
COUNTED_VALUES: list[tuple[str, str]] = [
    (r"^(\d+)_modules$",
     "The pipeline is spread over {0} source files, none of them dominant."),
    (r"^(\d+)_runnable_scripts$",
     "{0} files are runnable as programs. More than one means the documented "
     "single command is a sequence, or that several stages are invoked "
     "separately."),
]


def define(token: str, *, scope: str | None = None) -> str | None:
    """Definition of one canonical token.

    ``scope`` picks which vocabulary wins when a token exists in both — the
    solution-shape values and the methodology terms overlap on a few words
    ("random", "threshold") that mean different things in the two tables.
    """
    if not token:
        return None
    token = token.strip()
    first, second = ((VALUE_GLOSSARY, TERM_GLOSSARY) if scope != "term"
                     else (TERM_GLOSSARY, VALUE_GLOSSARY))
    for table in (first, second):
        if token in table:
            return table[token]
    for pattern, template in COUNTED_VALUES:
        m = re.match(pattern, token)
        if m:
            return template.format(*m.groups())
    return None


def split_value(value: str) -> list[str]:
    """``"npz_per_solve+pickle+index"`` -> its three tokens."""
    if not value:
        return []
    return [t for t in re.split(r"\+", str(value).strip()) if t]


def describe_value(value: str, *, scope: str | None = None) -> str:
    """Full prose for a (possibly composite) value, ready for hover text."""
    parts = []
    for token in split_value(value):
        d = define(token, scope=scope)
        parts.append(f"{token} — {d}" if d else f"{token} — (no definition recorded)")
    return "\n\n".join(parts)


def dimension_help(dim: str) -> str:
    """The question a cross-comparison row answers, and why it matters."""
    bits = [DIMENSION_QUESTIONS.get(dim, ""), DIMENSION_NOTES.get(dim, "")]
    return "\n\n".join(b for b in bits if b)


def all_terms() -> dict[str, dict[str, str]]:
    """Every defined token, grouped for a glossary section."""
    return {
        "How the prompt was solved (cross-comparison values)": dict(
            sorted(VALUE_GLOSSARY.items())),
        "Method and decision vocabulary": dict(sorted(TERM_GLOSSARY.items())),
    }


__all__ = ["all_terms", "define", "describe_value", "dimension_help", "split_value"]
