"""Full TokaMaker solve smoke test. Marked @pytest.mark.slow so default `pytest` skips it.

Run with: ``pytest tests/ -v -m slow``
"""

import pytest

from autotokamak.core.geometry import build_mesh_from_config
from autotokamak.core.solver import solve_equilibrium


@pytest.mark.slow
def test_smoke_solve_completes():
    cfg = {
        "boundary": {"npts": 80, "r0": 0.42, "z0": 0.0, "a": 0.15, "kappa": 1.4, "delta": 0.0},
        "mesh": {"regions": [{"name": "plasma", "type": "plasma", "dx": 0.015}]},
        "solver": {"order": 2, "F0": 0.10752, "maxits": 2, "free_boundary": False},
        "targets": {"Ip": 120000.0, "Ip_ratio": 1.0},
        "init_psi": {"method": "tokamaker_default"},
    }
    lcfs, _, mesh_pts, mesh_lc, mesh_reg = build_mesh_from_config(cfg)
    # We deliberately set maxits=2 to keep it fast; the solver will either converge
    # or return after 2 iterations. Either way it should not raise.
    gs = solve_equilibrium(
        mesh_pts=mesh_pts,
        mesh_lc=mesh_lc,
        mesh_reg=mesh_reg,
        lcfs=lcfs,
        cfg=cfg,
    )
    assert gs is not None
