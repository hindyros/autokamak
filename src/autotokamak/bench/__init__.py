# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Substrate-agnostic benchmark infrastructure.

Everything an agent-capability condition needs regardless of which harness
runs it: the run trace schema (``trace``), score dispatch (``scoring``), the
task specification (``taskspec``), deliverable-contract validation
(``contract``), and the ``python -m autotokamak.bench`` CLI.

Most reuse of this package starts with one of four things:

>>> from autotokamak.bench import validate_deliverables   # does an artifact satisfy the contract?
>>> from autotokamak.bench import score_against_frozen    # how good is it, on the frozen set?
>>> from autotokamak.bench import compute_diagnostics     # is it physically valid, and was the report honest?
>>> from autotokamak.bench import TaskSpec                # load and hash a task definition

``compute_diagnostics`` is the piece with the widest use outside this project:
it is what separates "the agent delivered the files" from "the artifact is
physically meaningful", and the two come apart far more often than one expects.
"""

from .contract import (
    rel_l2_errors,
    run_predict,
    score_against_frozen,
    validate_deliverables,
)
from .diagnostics import compute_diagnostics
from .taskspec import TaskSpec

__all__ = [
    "TaskSpec",
    "compute_diagnostics",
    "rel_l2_errors",
    "run_predict",
    "score_against_frozen",
    "validate_deliverables",
]
