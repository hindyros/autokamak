# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Tests for the solution-shape cross comparison (offline; no LLM, no solver).

Each fixture encodes a way this analysis was observed to go wrong on the real
corpus: a mandated smoke script deciding the campaign's execution model, a
masking helper with a house name, a "validation" that never re-loads anything.
"""
from __future__ import annotations

from autotokamak.bench.solution_shape import analyse_solution_shape, cross_compare


def _write(ws, rel, text):
    p = ws / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _dim(ws, name):
    return analyse_solution_shape(ws)["dimensions"][name]


def test_mandated_smoke_script_does_not_decide_the_execution_model(tmp_path):
    # Every workspace has a milestone/smoke script because the task demands
    # one. Reading the campaign's execution model off it is how this analysis
    # becomes fiction.
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws, "milestone_fixed_boundary.py",
           "import subprocess\n"
           "def main():\n"
           "    subprocess.run(['python', 'solve_one.py'])\n")
    _write(ws, "campaign.py",
           "from OpenFUSIONToolkit import OFT_env\n"
           "_ENV = None\n"
           "def get_env():\n"
           "    global _ENV\n"
           "    if _ENV is None:\n"
           "        _ENV = OFT_env(nthreads=1)\n"
           "    return _ENV\n"
           "def run_campaign(params):\n"
           "    for p in params:\n"
           "        solve(get_env(), p)\n")
    assert _dim(ws, "solve_isolation")["value"] == "in_process_serial"
    assert _dim(ws, "oft_env_strategy")["value"] == "singleton_reused"


def test_worker_process_ownership_is_distinguished_from_a_singleton(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws, "campaign.py",
           "from multiprocessing import Pool\n"
           "from OpenFUSIONToolkit import OFT_env\n"
           "def worker(p):\n"
           "    env = OFT_env(nthreads=1)\n"
           "    return solve(env, p)\n"
           "def run_campaign(params):\n"
           "    with Pool(4) as pool:\n"
           "        return pool.map(worker, params)\n")
    assert _dim(ws, "oft_env_strategy")["value"] == "per_worker_process"
    assert _dim(ws, "solve_isolation")["value"] == "process_pool"


def test_polygon_masking_is_recognised_under_a_house_name(tmp_path):
    # Agents ship _points_in_poly / inside_lcfs_mask / boundary_mask; the
    # operation is the same and the comparison must say so.
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws, "predict.py",
           "import numpy as np\n"
           "from model import boundary_mask\n"
           "def predict(params, R, Z):\n"
           "    psi = net(params)\n"
           "    m = boundary_mask(params, R, Z)\n"
           "    return np.where(m, psi, np.nan)\n")
    assert "lcfs_polygon" in _dim(ws, "mask_rule")["value"]


def test_storage_validation_requires_a_reload_not_a_claim(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws, "store.py",
           "import numpy as np\n"
           "def save_solve(path, psi):\n"
           "    np.savez(path, psi=psi)\n"
           "def verify(path):\n"
           "    d = np.load(path)\n"
           "    return np.isfinite(d['psi']).mean() > 0.01\n")
    assert _dim(ws, "storage_validation")["value"] == "reload_and_check_finite"

    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    _write(ws2, "store.py",
           "import numpy as np\n"
           "def save_solve(path, psi, log):\n"
           "    np.savez(path, psi=psi)\n"
           "    log.write({'stored_ok': True, 'finite_frac': 0.2})\n")
    assert _dim(ws2, "storage_validation")["value"] == "claimed_only"


def test_self_test_is_found_in_the_verification_script(tmp_path):
    # The deliverable self-test lives, by design, in a repro/check script —
    # exactly the kind of file excluded from the production scope.
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws, "run_pipeline.py", "def main():\n    train()\n")
    _write(ws, "final_repro_check.py",
           "import subprocess, sys\n"
           "def check():\n"
           "    subprocess.run([sys.executable, 'predict.py', '--input', 'p.json',\n"
           "                    '--output', 'o.npz'], check=True)\n")
    assert _dim(ws, "self_test")["value"] == "subprocess_reruns_predict"


def test_hand_built_mesh_is_flagged_as_the_forbidden_route(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws, "mesh.py",
           "from scipy.spatial import Delaunay\n"
           "def build(points):\n"
           "    return Delaunay(points)\n")
    d = _dim(ws, "mesh_route")
    assert d["value"] == "hand_built_triangulation"
    assert "forbidden" in (d["note"] or "")


def test_leakage_check_is_function_scoped(tmp_path):
    # A single-file pipeline mentions "test" somewhere by necessity; only a
    # reference inside a fitting function is worth reading.
    ws = tmp_path / "ws"
    ws.mkdir()
    _write(ws, "pipeline.py",
           "def make_test_set(seed):\n"
           "    return draw(seed)\n"
           "def fit_model(X, y):\n"
           "    model.fit(X, y)\n"
           "    return model\n")
    assert _dim(ws, "leakage_guard")["value"] == "test_absent_from_fitting_functions"

    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    _write(ws2, "pipeline.py",
           "def fit_model(X, y, X_test):\n"
           "    model.fit(X, y)\n"
           "    score = model.score(X_test)\n"
           "    return model, score\n")
    assert _dim(ws2, "leakage_guard")["value"] == "test_referenced_in_fitting_function"


def test_cross_compare_transposes_and_ranks_disagreement():
    a = {"dimensions": {"mask_rule": {"value": "lcfs_polygon", "evidence": []},
                        "storage": {"value": "hdf5", "evidence": []}}}
    b = {"dimensions": {"mask_rule": {"value": "lcfs_polygon", "evidence": []},
                        "storage": {"value": "npz_per_solve", "evidence": []}}}
    rows = {r["dimension"]: r for r in cross_compare({"A": a, "B": b})}
    assert rows["mask_rule"]["A"] == rows["mask_rule"]["B"] == "lcfs_polygon"
    assert rows["mask_rule"]["agreement"] == "all_same"
    assert rows["storage"]["agreement"] == "all_differ"
    # Dimensions neither run evidenced still get a row, marked unknown.
    assert rows["pilot_gate"]["A"] == "-"


def test_empty_workspace_never_raises(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    out = analyse_solution_shape(ws)
    assert out["dimensions"] == {}
    assert out["n_files"] == 0
