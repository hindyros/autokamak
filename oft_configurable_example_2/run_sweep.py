#!/usr/bin/env python3
"""Run a discretization sweep for the config-driven OFT/TokaMaker equilibrium example.

Reads discretization_sweep.yaml, deep-merges each case's overrides into the baseline
config, writes derived configs into sweep directory, and runs run_equilibrium.py
for each case.

Constraints: uses only stdlib + PyYAML (already used by run_equilibrium.py).
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def deep_merge(a: dict, b: dict) -> dict:
    """Return deep-merged dict: values from b override/extend a."""
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="discretization_sweep.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sweep_path = Path(args.sweep)
    sweep = yaml.safe_load(sweep_path.read_text())

    baseline_path = Path(sweep["baseline_config"])
    baseline = yaml.safe_load(baseline_path.read_text())

    auto = sweep.get("automation", {})
    derived_dir = Path(auto.get("derived_config_dir", "sweeps"))
    derived_dir.mkdir(parents=True, exist_ok=True)

    cmd_tmpl = auto.get("run_command_template", "python run_equilibrium.py --config {config}")

    results = []
    for case in sweep.get("cases", []):
        cid = case["id"]
        overrides = case.get("overrides", {})

        cfg = deep_merge(baseline, overrides)
        # Ensure output case name follows sweep id unless explicitly overridden
        cfg.setdefault("case", {})
        cfg["case"].setdefault("name", cid)

        out_cfg = derived_dir / f"{cid}.yaml"
        out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))

        cmd = cmd_tmpl.format(config=str(out_cfg))
        print(f"\n=== Case: {cid} ===")
        print(f"Config: {out_cfg}")
        print(f"Cmd: {cmd}")

        if args.dry_run:
            results.append((cid, "DRY_RUN", 0.0))
            continue

        t0 = time.time()
        p = subprocess.run(cmd, shell=True)
        dt = time.time() - t0
        results.append((cid, "OK" if p.returncode == 0 else f"FAIL({p.returncode})", dt))

    print("\n=== Sweep summary ===")
    for cid, status, dt in results:
        print(f"{cid:24s} {status:10s} {dt:8.2f} s")

    # nonzero exit if any failures
    if any("FAIL" in s for _, s, _ in results):
        sys.exit(2)


if __name__ == "__main__":
    main()
