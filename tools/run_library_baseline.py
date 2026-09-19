#!/usr/bin/env python
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Library baseline: what does `autotokamak` itself score, with no agent at all?

The matrix-v4 campaign found that agents given the library (L2) did WORSE
than agents writing from scratch (L3), in all four substrates. The obvious
objection is the one a reviewer will raise first:

    "Maybe your library is simply worse than what a competent agent writes."

That is answerable without a single API call. L0 runs the same library on the
same problem with scripted, seeded decisions and no LLM, so this arm puts a
number on the library's own ceiling:

  * if L0 lands near the best L3 runs, the library is not the handicap and
    the L2 deficit is about how agents USE it;
  * if L0 lands near the L2 runs, the library IS the ceiling and the paper's
    claim has to be worded about this library rather than about
    library-assistance in general.

Budget matters as much as the score. The committed L0 dataset holds 2000
solves — four times what the agents were allowed — so an unmatched baseline
would flatter the library and prove nothing. Runs here are MATCHED to the
task's campaign bound by subsampling the dataset (default 450 successful
solves); pass --n-samples 0 to also see the unlimited-data ceiling.

Scored on the same frozen test set, through the same relative-L2, as every
agent cell. Zero API spend; solver/CPU time only.

Usage:
    python tools/run_library_baseline.py --seeds 0 1 2 --n-samples 450
    python tools/run_library_baseline.py --seeds 0 --n-samples 0   # unlimited
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

SOURCE_DATASET = REPO_ROOT / "examples" / "dataset_generation" / "L0" / "outputs" / "dataset.h5"
META_OUTPUT = REPO_ROOT / "examples" / "surrogate_meta" / "L0"


def subsample(source: Path, n: int, seed: int, dest: Path) -> int:
    """A budget-matched copy of the dataset: n SUCCESSFUL solves, seeded.

    Matching on successful solves rather than rows is the honest comparison:
    the agents' budget was spent on attempts, but what either side gets to
    train on is what converged.
    """
    import numpy as np

    from autotokamak.data.h5io import read_h5_arrays, write_h5_arrays

    a = read_h5_arrays(source)
    ok = np.flatnonzero(np.asarray(a.success, dtype=bool))
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(ok, size=min(n, ok.size), replace=False))
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_h5_arrays(dest, a.take(take))
    return int(take.size)


def score_on_frozen(workspace: Path) -> dict:
    """Same yardstick as every agent cell: relative-L2 on the frozen set."""
    import numpy as np

    from matrix_report import baseline_errors, load_frozen, predict_meta_cell, rel_l2

    frozen = load_frozen()
    pred = predict_meta_cell(workspace, frozen)
    errs = rel_l2(pred, frozen["psi"])
    base = baseline_errors(frozen)
    finite = errs[np.isfinite(errs)]
    return {
        "test_rel_l2": {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "p90": float(np.percentile(finite, 90)),
            "n": int(finite.size),
        },
        "baseline_rel_l2": {"mean": float(np.nanmean(base))},
        "accuracy_pct": float(100 * (1 - np.mean(finite) / np.nanmean(base))),
    }


def run_one(seed: int, n_samples: int, out_root: Path, iterations: int,
            time_budget: int) -> dict:
    label = f"seed{seed}_n{n_samples or 'all'}"
    run_dir = out_root / label
    if run_dir.exists():
        shutil.rmtree(run_dir)

    dataset = SOURCE_DATASET
    n_used = None
    if n_samples:
        dataset = out_root / "data" / f"dataset_{label}.h5"
        n_used = subsample(SOURCE_DATASET, n_samples, seed, dataset)

    cmd = [sys.executable, "-m", "autotokamak.pipelines", "meta",
           "--level", "L0", "--seed", str(seed),
           "--dataset", str(dataset),
           "--max-iterations", str(iterations),
           "--time-budget", str(time_budget)]
    print(f"[baseline] {label}: {' '.join(cmd)}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    wall = round(time.time() - started, 1)
    if proc.returncode != 0:
        return {"label": label, "seed": seed, "n_train_samples": n_used,
                "error": proc.stderr[-800:], "wall_s": wall}

    # The pipeline writes to a fixed workspace; move it aside so the next
    # seed cannot overwrite it and so every run keeps its own artifacts.
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(META_OUTPUT), str(run_dir))

    row: dict = {"label": label, "seed": seed, "n_train_samples": n_used,
                 "wall_s": wall, "workspace": str(run_dir)}
    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        m = json.loads(manifest.read_text())
        row.update({k: m.get(k) for k in
                    ("n_iterations", "terminated_by", "winner_model_name",
                     "final_accuracy_pct", "final_rmse")})
    try:
        row["frozen_score"] = score_on_frozen(run_dir)
    except Exception as exc:  # noqa: BLE001
        row["frozen_score"] = {"error": f"{type(exc).__name__}: {exc}"}
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-samples", type=int, default=450,
                    help="Successful solves to train on (0 = the full 2000, "
                         "i.e. the unlimited-data ceiling)")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--time-budget", type=int, default=600,
                    help="Phase-2 Optuna budget per iteration, seconds")
    ap.add_argument("--tag", default="library-baseline")
    args = ap.parse_args()

    out_root = REPO_ROOT / "experiments" / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "library_baseline.json"
    rows = []
    if results_path.is_file():
        try:
            rows = json.loads(results_path.read_text())
        except Exception:  # noqa: BLE001
            rows = []

    for seed in args.seeds:
        row = run_one(seed, args.n_samples, out_root, args.iterations,
                      args.time_budget)
        rows.append(row)
        results_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[baseline] {json.dumps(row.get('frozen_score', {}))}", flush=True)

    print("\n=== LIBRARY BASELINE (no agent, no LLM) ===")
    print(f"{'run':<16}{'n_train':>9}{'winner':>14}{'rel_l2':>10}{'acc %':>8}")
    for r in rows:
        mean = ((r.get("frozen_score") or {}).get("test_rel_l2") or {}).get("mean")
        acc = r.get("final_accuracy_pct")
        print(f"{r['label']:<16}"
              f"{str(r.get('n_train_samples') or 'all'):>9}"
              f"{str(r.get('winner_model_name')):>14}"
              f"{(f'{mean:.4f}' if isinstance(mean, float) else '-'):>10}"
              f"{(f'{acc:.1f}' if isinstance(acc, float) else '-'):>8}")
    print(f"\nWrote {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
