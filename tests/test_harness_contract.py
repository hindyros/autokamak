# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Every registered harness must satisfy the adapter contract.

Before this file, five of the six adapters were never instantiated by any test.
A contributor adding a substrate could get all the way to a paid campaign
before discovering that their class forgot to declare `name`, because the base
class annotates it without a default and the failure surfaces deep inside
`condition_for`.

These checks are cheap and offline: they instantiate each adapter and inspect
it. Nothing here runs an agent or touches a provider.
"""
from __future__ import annotations

import inspect

import pytest

from autotokamak.bench.taskspec import TaskSpec
from autotokamak.harnesses import HARNESS_NAMES, Harness, get_harness


@pytest.fixture
def task():
    return TaskSpec(task_id="contract-probe", access_level="L3",
                    problem="a placeholder problem statement")


@pytest.mark.parametrize("name", HARNESS_NAMES)
def test_harness_is_importable_and_declares_itself(name):
    """The adapter loads, subclasses Harness, and its name matches the registry."""
    harness = get_harness(name)
    assert isinstance(harness, Harness)
    assert getattr(type(harness), "name", None) == name, (
        f"{type(harness).__name__} must set `name = {name!r}` as a ClassVar; "
        "the base class only annotates it, so a missing one fails much later"
    )


@pytest.mark.parametrize("name", HARNESS_NAMES)
def test_harness_run_has_the_agreed_signature(name):
    """`run` must accept the arguments the driver passes, keyword-only where required."""
    sig = inspect.signature(get_harness(name).run)
    assert list(sig.parameters)[:2] == ["task", "workspace"]
    for kw in ("run_dir", "model", "timeout_seconds"):
        assert kw in sig.parameters, f"{name}.run is missing {kw!r}"
        assert sig.parameters[kw].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name}.run must take {kw!r} keyword-only, as the driver passes it that way"
        )


@pytest.mark.parametrize("name", HARNESS_NAMES)
def test_harness_is_documented(name):
    """An adapter is the first thing a contributor reads; it needs to say what it is."""
    cls = type(get_harness(name))
    assert (cls.__doc__ or "").strip(), f"{cls.__name__} has no docstring"


@pytest.mark.parametrize("name", HARNESS_NAMES)
def test_dry_run_info_describes_the_run_without_making_it(name, task, tmp_path):
    """`dry_run_info` backs `bench run --dry-run`, which must never execute anything."""
    info = get_harness(name).dry_run_info(task, tmp_path)
    assert isinstance(info, dict)
    assert info.get("harness") == name
    assert not list(tmp_path.iterdir()), "a dry run must not write to the workspace"


def test_registry_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown harness"):
        get_harness("not-a-harness")
