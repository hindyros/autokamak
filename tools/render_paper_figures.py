#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Render the paper's figures from campaign artifacts only.

Every point plotted is read from `experiments/<tag>/` -- `result.json` per run
and `cost_report.csv` for dollars. Nothing is recomputed, smoothed, or
hand-placed, so a figure cannot drift away from the table it sits next to.

    python tools/render_paper_figures.py --tag matrix-v4-20260919 \
           --out docs/paper/neurips_benchmark/figures

Writes three PDFs (vector, Type-1 fonts -- venues reject Type 3):

  fig_levels.pdf        every scored run's relative-L2, by cell, with the
                        four within-harness pairs drawn as connectors.
  fig_verification.pdf  honesty gap against independently scored error, marked
                        by whether the run passed every gate while being
                        physically invalid.
  fig_cost.pdf          dollars against accuracy per 100 solver calls, per cell.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed, never Type 3
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 200,
})
import matplotlib.pyplot as plt  # noqa: E402

HARNESSES = ["cursor", "pi", "dspy", "ursa"]     # ordered by median error
LEVEL_COLOUR = {"L2": "#B8860B", "L3": "#1F4E79"}
BASELINE = 0.49334822264351563                    # mean-flux-map predictor


def load_runs(tag_dir: Path) -> list[dict]:
    rows = []
    for rj in sorted(tag_dir.glob("*/*/result.json")):
        r = json.loads(rj.read_text())
        diag = r.get("diagnostics") or {}
        rel = ((r.get("frozen_score") or {}).get("test_rel_l2") or {}).get("mean")
        cond = rj.parent.parent.name
        level, _, harness = cond.partition("-")
        rows.append({
            "condition": cond, "level": level, "harness": harness,
            "run_dir": rj.parent.name, "status": r.get("status"),
            "passed": bool((r.get("contract") or {}).get("passed")),
            "rel_l2": rel,
            "valid": diag.get("physically_valid"),
            "invalid_but_passed": bool(diag.get("passed_gates_but_invalid")),
            "honesty_gap": diag.get("honesty_gap"),
        })
    return rows


def load_costs(tag_dir: Path) -> dict[str, float]:
    """Per-cell dollars from cost_report.csv, which keys on the directory and
    so does not suffer the run_id collision that corrupts aggregate.csv."""
    totals: dict[str, float] = {}
    path = tag_dir / "cost_report.csv"
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("cost_usd"):
                totals[row["condition"]] = (totals.get(row["condition"], 0.0)
                                            + float(row["cost_usd"]))
    return totals


def load_cells(tag_dir: Path) -> dict[str, dict]:
    with (tag_dir / "aggregate.csv").open() as fh:
        return {r["condition"]: r for r in csv.DictReader(fh)}


def fig_levels(runs, cells, out: Path) -> None:
    """Per-cell error with the paired level contrast drawn explicitly."""
    fig, ax = plt.subplots(figsize=(5.4, 2.0))
    rng = __import__("numpy").random.default_rng(20260917)
    xticks, xlabels = [], []
    for i, h in enumerate(HARNESSES):
        medians = {}
        for j, lvl in enumerate(("L2", "L3")):
            x = i * 1.0 + (j - 0.5) * 0.30
            pts = [r["rel_l2"] for r in runs
                   if r["harness"] == h and r["level"] == lvl and r["rel_l2"]]
            ax.scatter(x + rng.uniform(-0.055, 0.055, len(pts)), pts, s=11,
                       facecolor="none", edgecolor=LEVEL_COLOUR[lvl],
                       linewidth=0.7, zorder=3)
            med = float(cells[f"{lvl}-{h}"]["rel_l2_median"])
            medians[lvl] = med
            ax.plot([x - 0.11, x + 0.11], [med, med],
                    color=LEVEL_COLOUR[lvl], linewidth=1.8, zorder=4)
            lo = cells[f"{lvl}-{h}"]["rel_l2_ci_lo"]
            hi = cells[f"{lvl}-{h}"]["rel_l2_ci_hi"]
            if lo and hi:
                ax.plot([x, x], [float(lo), float(hi)],
                        color=LEVEL_COLOUR[lvl], linewidth=0.8, alpha=0.65, zorder=2)
            xticks.append(x)
            xlabels.append(lvl)
        ax.annotate("", xy=(i + 0.15, medians["L3"]), xytext=(i - 0.15, medians["L2"]),
                    arrowprops=dict(arrowstyle="->", color="0.35",
                                    linewidth=0.7, shrinkA=2, shrinkB=2))
        ax.text(i, 3.2, h, ha="center", va="top", fontsize=8)
        ax.text(i, min(medians.values()) * 0.45,
                f"{medians['L3'] - medians['L2']:+.3f}",
                ha="center", va="center", fontsize=6.5, color="0.3",
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec="none", alpha=0.85))
    ax.axhline(BASELINE, color="crimson", linewidth=0.8, linestyle=(0, (4, 2)))
    ax.text(-0.52, BASELINE * 1.09, "mean-map baseline", fontsize=6.5,
            color="crimson", ha="left")
    ax.set_yscale("log")
    ax.set_ylim(0.012, 4.2)
    ax.set_xlim(-0.55, 3.55)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(r"relative $L_2$  (lower is better)")
    handles = [plt.Line2D([], [], color=LEVEL_COLOUR[k], linewidth=1.8,
                          label={"L2": "L2 library-assisted",
                                 "L3": "L3 from scratch"}[k])
               for k in ("L2", "L3")]
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "fig_levels.pdf")
    plt.close(fig)


def fig_verification(runs, out: Path) -> None:
    """Delivery, validity and honesty are three different questions."""
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    groups = [
        ([r for r in runs if r["rel_l2"] and r["honesty_gap"] is not None
          and not r["invalid_but_passed"]],
         dict(marker="o", s=13, facecolor="none", edgecolor="#1F4E79",
              linewidth=0.7, label="passed, valid")),
        ([r for r in runs if r["rel_l2"] and r["honesty_gap"] is not None
          and r["invalid_but_passed"]],
         dict(marker="X", s=22, color="#C0392B", linewidth=0,
              label="passed all gates, invalid")),
    ]
    for rows, style in groups:
        ax.scatter([r["rel_l2"] for r in rows], [r["honesty_gap"] for r in rows],
                   zorder=3, **style)
    ax.axhline(0, color="0.5", linewidth=0.7)
    ax.axvline(BASELINE, color="crimson", linewidth=0.8, linestyle=(0, (4, 2)))
    ax.text(BASELINE * 1.06, 1.55, "no better than\nthe mean map", fontsize=6,
            color="crimson", va="top")
    ax.set_xscale("log")
    ax.set_xlabel(r"independently scored relative $L_2$")
    ax.set_ylabel("honesty gap  (self $-$ scored)")
    ax.text(0.022, -0.78, "claimed better\nthan it was", fontsize=6, color="0.3")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "fig_verification.pdf")
    plt.close(fig)


def fig_cost(cells, costs, out: Path) -> None:
    """What each cell's accuracy cost, at one model."""
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for cond, row in cells.items():
        lvl, _, h = cond.partition("-")
        usd, acc = costs.get(cond), float(row["acc_per_100_solves"])
        if not usd:
            continue
        ax.scatter(usd, acc, s=26, marker="o" if lvl == "L2" else "^",
                   facecolor="none" if lvl == "L2" else LEVEL_COLOUR[lvl],
                   edgecolor=LEVEL_COLOUR[lvl], linewidth=0.9, zorder=3)
        ax.annotate(h, (usd, acc), textcoords="offset points", xytext=(4, 3),
                    fontsize=6.5, color=LEVEL_COLOUR[lvl])
    ax.axhline(0, color="0.5", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlim(2.8, 140)
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("cell cost (USD, log scale)")
    ax.set_ylabel("accuracy per 100 solver calls")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor=LEVEL_COLOUR["L2"], label="L2 library-assisted"),
        plt.Line2D([], [], marker="^", linestyle="", color=LEVEL_COLOUR["L3"],
                   label="L3 from scratch"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "fig_cost.pdf")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiments-dir", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    tag_dir = (Path(args.experiments_dir) if args.experiments_dir
               else root / "experiments") / args.tag
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs, cells, costs = load_runs(tag_dir), load_cells(tag_dir), load_costs(tag_dir)
    fig_levels(runs, cells, out)
    fig_verification(runs, out)
    fig_cost(cells, costs, out)
    n = sum(1 for r in runs if r["rel_l2"])
    print(f"Wrote 3 figures to {out} from {n} scored runs of {len(runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
