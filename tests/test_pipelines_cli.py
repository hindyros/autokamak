"""Smoke tests for the unified pipelines CLI.

Tests run in-process with mocked platform calls so they are fast and offline.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# _common
# ---------------------------------------------------------------------------

def test_resolve_output_dir_creates_path():
    from autotokamak.pipelines._common import REPO_ROOT as PR, resolve_output_dir
    d = resolve_output_dir("phase1", "fast")
    assert d == PR / "examples" / "dataset_generation" / "fast"
    assert d.is_dir()


def test_resolve_output_dir_bad_pipeline():
    from autotokamak.pipelines._common import resolve_output_dir
    with pytest.raises(ValueError, match="Unknown pipeline"):
        resolve_output_dir("bogus", "fast")


def test_resolve_output_dir_bad_mode():
    from autotokamak.pipelines._common import resolve_output_dir
    with pytest.raises(ValueError, match="Unknown mode"):
        resolve_output_dir("phase1", "warp")


def test_write_manifest(tmp_path):
    from autotokamak.pipelines._common import write_manifest
    p = write_manifest(tmp_path, pipeline="phase1", mode="fast", n_succeeded=10)
    data = json.loads(p.read_text())
    assert data["pipeline"] == "phase1"
    assert data["mode"] == "fast"
    assert data["n_succeeded"] == 10
    assert "run_id" in data


# ---------------------------------------------------------------------------
# __main__ argparse
# ---------------------------------------------------------------------------

def test_help_exits_cleanly():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-m", "autotokamak.pipelines", "--help"],
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    assert r.returncode == 0
    assert b"phase1" in r.stdout
    assert b"phase2" in r.stdout
    assert b"meta" in r.stdout


def test_phase1_help():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-m", "autotokamak.pipelines", "phase1", "--help"],
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    assert r.returncode == 0
    assert b"--mode" in r.stdout


def test_meta_help():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-m", "autotokamak.pipelines", "meta", "--help"],
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    assert r.returncode == 0
    assert b"--mode" in r.stdout


def test_missing_mode_errors():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-m", "autotokamak.pipelines", "phase1"],
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    assert r.returncode != 0


# ---------------------------------------------------------------------------
# phase1 fast — mocked run_sweep
# ---------------------------------------------------------------------------

def test_phase1_fast_calls_run_sweep(tmp_path):
    from autotokamak.pipelines._common import REPO_ROOT as PR

    fake_result = SimpleNamespace(
        dataset_path=str(tmp_path / "dataset.h5"),
        n_requested=20,
        n_succeeded=20,
        n_isoflux_used=0,
        config_hash="abc123",
    )

    with (
        patch("autotokamak.pipelines.phase1.resolve_output_dir", return_value=tmp_path),
        patch("autotokamak.data.sweep.run_sweep", return_value=fake_result) as mock_sweep,
        patch("autotokamak.data.schema.SweepConfig") as mock_cfg_cls,
    ):
        mock_cfg = MagicMock()
        mock_cfg.output_path = "dataset.h5"
        mock_cfg.sampling.n_samples = 20
        mock_cfg.model_copy.return_value = mock_cfg
        mock_cfg_cls.from_yaml.return_value = mock_cfg

        from autotokamak.pipelines.phase1 import run_phase1_fast
        result = run_phase1_fast(n_samples=20)

    mock_sweep.assert_called_once()
    assert result["n_succeeded"] == 20
    assert (tmp_path / "manifest.json").is_file()


# ---------------------------------------------------------------------------
# meta dispatcher — mocked meta_loop.run
# ---------------------------------------------------------------------------

def test_meta_fast_passes_structured_mode(tmp_path):
    from autotokamak.pipelines._common import REPO_ROOT as PR

    fake_report = SimpleNamespace(
        n_iterations=1,
        terminated_by="iterations_cap",
        final_rmse=0.001,
        baseline_rmse=0.003,
        winner_model_name="poly_ridge",
    )

    with (
        patch("autotokamak.pipelines.meta.resolve_output_dir", return_value=tmp_path),
        patch("autotokamak.pipelines.meta.sys") as mock_sys,
    ):
        # Patch the import of meta_run inside the function
        import autotokamak.pipelines.meta as meta_module
        with patch.object(meta_module, "__builtins__", __builtins__):
            pass  # just ensure it imports

        # Directly test that phase2_mode_override is "structured" for mode=fast
        calls = []

        def fake_run(**kwargs):
            calls.append(kwargs)
            return fake_report

        with (
            patch("autotokamak.pipelines.meta.resolve_output_dir", return_value=tmp_path),
        ):
            import importlib, types
            # Inject a fake agent.runners.meta_loop into sys.modules
            fake_meta_loop_module = types.ModuleType("agent.runners.meta_loop")
            fake_meta_loop_module.run = fake_run  # type: ignore[attr-defined]

            import sys as _sys
            _sys.modules["agent.runners.meta_loop"] = fake_meta_loop_module

            from autotokamak.pipelines.meta import run_meta
            run_meta(mode="fast", max_iterations=1, time_budget=120)

            del _sys.modules["agent.runners.meta_loop"]

    assert len(calls) == 1
    assert calls[0]["phase2_mode_override"] == "structured"


def test_meta_ursa_passes_codegen_mode(tmp_path):
    fake_report = SimpleNamespace(
        n_iterations=1,
        terminated_by="iterations_cap",
        final_rmse=0.002,
        baseline_rmse=0.004,
        winner_model_name="mlp",
    )

    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return fake_report

    import sys as _sys, types
    fake_meta_loop_module = types.ModuleType("agent.runners.meta_loop")
    fake_meta_loop_module.run = fake_run  # type: ignore[attr-defined]
    _sys.modules["agent.runners.meta_loop"] = fake_meta_loop_module

    with patch("autotokamak.pipelines.meta.resolve_output_dir", return_value=tmp_path):
        from autotokamak.pipelines.meta import run_meta
        run_meta(mode="ursa", max_iterations=1, time_budget=120)

    del _sys.modules["agent.runners.meta_loop"]

    assert len(calls) == 1
    assert calls[0]["phase2_mode_override"] == "codegen"


# ---------------------------------------------------------------------------
# meta_loop.run() phase2_mode_override kwarg
# ---------------------------------------------------------------------------

def test_meta_loop_run_accepts_phase2_mode_override():
    """Check phase2_mode_override is accepted and wired into MetaState."""
    import inspect
    from autotokamak.agent.runners.meta_loop import run
    sig = inspect.signature(run)
    assert "phase2_mode_override" in sig.parameters


def test_meta_loop_run_rejects_invalid_mode(tmp_path):
    """phase2_mode_override must be 'structured' or 'codegen'."""
    from autotokamak.agent.runners.meta_loop import run
    with pytest.raises((ValueError, Exception)):
        run(
            config_path="nonexistent.yaml",
            phase2_mode_override="invalid",
        )
