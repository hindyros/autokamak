#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Physics figures for the paper: what the agents were asked to predict, and what
they actually produced.

The paper had no picture of its own subject. A reader who has never seen a
tokamak met the Grad-Shafranov equation on page one and had to take on faith
both what psi is and what a relative-L2 of 0.07 versus 0.99 looks like. These
figures answer both by showing the field.

Three stages, each cached so the later ones are cheap:

  cache    run each selected agent's predict.py against the frozen 60-case test
           set and store the (60, 96, 64) array. The benchmark never persisted
           these -- scoring kept only mean/median/p90 and dropped the field --
           so without this step the pictures cannot be drawn at all.
  render   draw the figures from the cache.
  all      both (default).

    python tools/render_physics_figures.py --tag matrix-v4-20260919 \
           --out docs/paper/neurips_benchmark/figures

Caching is offline and free: run_predict strips every API_KEY/SECRET/TOKEN from
the environment, and no predict.py in this corpus runs a Grad-Shafranov solve.
Each workspace is copied to a scratch directory first, because run_predict
creates its temp dir *inside* the workspace and predict.py writes __pycache__ --
and agent-generated output is not ours to modify.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

# The mean-flux-map predictor, i.e. "no model at all". Every accuracy figure in
# the paper is quoted against it.
BASELINE_REL_L2 = 0.49334822264351563


# ---------------------------------------------------------------------------
# Which runs to draw, and why each one is in the figure
# ---------------------------------------------------------------------------

# label -> (condition, run_id, one-line reason it earns a panel)
SELECTED: dict[str, tuple[str, str, str]] = {
    "best":        ("L3-ursa",   "20260919T050413Z-2", "best in the campaign, relL2 0.019"),
    "good":        ("L3-cursor", "20260919T050413Z",   "a typical good from-scratch run"),
    "unmasked":    ("L3-dspy",   "20260919T050413Z",   "85% accurate inside, fills the vacuum outside"),
    "typical_l3":  ("L3-dspy",   "20260919T215725Z",   "median-ish L3"),
    "typical_l2":  ("L2-cursor", "20260919T055817Z",   "median-ish L2"),
    "l2_pi":       ("L2-pi",     "20260919T062656Z",   "library-assisted, PCA + kernel ridge"),
    "constant":    ("L3-ursa",   "20260919T213658Z-3", "correct mask, almost no variation"),
    "catastrophic":("L2-ursa",   "20260920T040328Z",   "relL2 2.49, spread 16x ground truth"),
}


# ---------------------------------------------------------------------------
# Stage 1 -- cache predictions
# ---------------------------------------------------------------------------

def cache_predictions(tag_dir: Path, cache_dir: Path, *, force: bool = False) -> dict[str, Path]:
    """Run each selected agent's predict.py on the frozen set and store the field.

    Returns label -> npz path. Each npz holds ``psi`` (60, 96, 64) plus the run's
    identity, so the figures regenerate later without a working predictor --
    which matters, because several L2 workspaces depend on scikit-learn pickles
    that a future environment may refuse to load.
    """
    from matrix_report import load_frozen, predict_bench_cell

    frozen = load_frozen()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    for label, (cond, run_id, why) in SELECTED.items():
        dest = cache_dir / f"{cond}__{run_id}.npz"
        out[label] = dest
        if dest.exists() and not force:
            print(f"[cached] {label:13s} {cond}/{run_id}")
            continue
        ws = tag_dir / cond / run_id / "workspace"
        if not (ws / "predict.py").is_file():
            print(f"[skip]   {label:13s} {cond}/{run_id} — no predict.py", file=sys.stderr)
            continue

        print(f"[predict] {label:13s} {cond}/{run_id} ({why})", flush=True)
        with tempfile.TemporaryDirectory(prefix="psi_pred_") as tmp:
            # symlinks=True so the workspace's OpenFUSIONToolkit link is copied
            # as a link rather than duplicating the whole toolkit.
            scratch = Path(tmp) / "workspace"
            shutil.copytree(ws, scratch, symlinks=True)
            psi = predict_bench_cell(scratch, frozen)

        np.savez_compressed(dest, psi=psi.astype(np.float32),
                            condition=cond, run_id=run_id, reason=why)
        print(f"          -> {dest.name}  {psi.shape}")
    return out


def load_cached(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=False)["psi"], dtype=np.float64)


# ---------------------------------------------------------------------------
# Stage 2 -- figures
# ---------------------------------------------------------------------------

def _style():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "figure.dpi": 200,
    })
    import matplotlib.pyplot as plt
    return plt


def _psi_panel(ax, R, Z, field, *, vmin, vmax, cmap="RdBu_r", contours=True,
               lcfs=None, title=None):
    """One flux-field panel, drawn the way the plasma literature expects."""
    pc = ax.pcolormesh(R, Z, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    if contours and np.isfinite(field).any():
        ax.contour(R, Z, field, levels=np.linspace(vmin, vmax, 12),
                   colors="k", linewidths=0.3, alpha=0.45)
    if lcfs is not None:
        ax.plot(lcfs[:, 0], lcfs[:, 1], "k-", linewidth=0.9)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    if title:
        ax.set_title(title, fontsize=7.5, pad=3)
    return pc


def _lcfs_for(record: dict) -> np.ndarray:
    from autotokamak.core.geometry import build_lcfs
    return build_lcfs(r0=record["r0"], z0=0.0, a=record["a"],
                      kappa=record["kappa"], delta=record["delta"], npts=200)


def fig_primer(frozen, out: Path) -> None:
    """What a tokamak equilibrium is, for a reader who has never seen one."""
    plt = _style()
    from autotokamak.core.geometry import build_lcfs

    fig = plt.figure(figsize=(6.6, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.25], wspace=0.40)

    # (a) the shape and the five parameters that set it
    ax = fig.add_subplot(gs[0])
    r0, a, kappa, delta = 0.45, 0.16, 1.4, 0.35
    b = build_lcfs(r0=r0, z0=0.0, a=a, kappa=kappa, delta=delta, npts=300)
    ax.fill(b[:, 0], b[:, 1], color="#d8e6f3", zorder=0)
    ax.plot(b[:, 0], b[:, 1], "k-", lw=1.2, zorder=3)
    ax.plot([0, 0], [-0.30, 0.34], color="0.6", lw=0.8, ls=(0, (5, 3)))
    ax.text(-0.018, 0.0, "axis of symmetry", fontsize=5.8, color="0.45",
            rotation=90, va="center", ha="center")
    ax.annotate("", xy=(r0, 0), xytext=(0, 0), zorder=4,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#B8860B"))
    ax.text(r0 / 2, 0.016, r"$r_0$", color="#B8860B", fontsize=8, ha="center", zorder=5)
    ax.annotate("", xy=(r0 + a, 0), xytext=(r0, 0), zorder=4,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#1F4E79"))
    ax.text(r0 + a / 2, -0.042, r"$a$", color="#1F4E79", fontsize=8, ha="center", zorder=5)
    ax.annotate("", xy=(r0, a * kappa), xytext=(r0, 0), zorder=4,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#1F4E79"))
    ax.text(r0 + 0.014, a * kappa / 2, r"$\kappa a$", color="#1F4E79", fontsize=8, zorder=5)
    top = b[np.argmax(b[:, 1])]
    ax.annotate("", xy=(top[0], top[1] + 0.035), xytext=(r0, top[1] + 0.035), zorder=4,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#7a3b8f"))
    ax.text(r0 - a * delta - 0.03, top[1] + 0.030, r"$\delta a$", color="#7a3b8f",
            fontsize=8, ha="right", va="center", zorder=5)
    ax.set_xlim(-0.06, 0.70); ax.set_ylim(-0.31, 0.33)
    ax.set_aspect("equal"); ax.set_xlabel("$R$ [m]"); ax.set_ylabel("$Z$ [m]")
    ax.set_title("(a) the plasma boundary", fontsize=8)
    for s_ in ax.spines.values():
        s_.set_linewidth(0.4)

    # (b) a real solved field from the frozen test set
    ax = fig.add_subplot(gs[1])
    i = int(np.argmin(np.abs(frozen["X"][:, 2] - 1.3)))
    psi = frozen["psi"][i]
    vmin, vmax = float(np.nanmin(psi)), float(np.nanmax(psi))
    pc = _psi_panel(ax, frozen["R"], frozen["Z"], psi, vmin=vmin, vmax=vmax,
                    lcfs=_lcfs_for(frozen["records"][i]))
    ax.set_title("(b) the solved flux", fontsize=8)
    ax.set_xlabel("$R$ [m]"); ax.set_ylabel("$Z$ [m]")
    cb = fig.colorbar(pc, ax=ax, shrink=0.86, pad=0.04)
    cb.set_label(r"$\psi$ [Wb]", fontsize=7); cb.ax.tick_params(labelsize=6)

    # (c) what elongation and triangularity do to the shape
    ax = fig.add_subplot(gs[2])
    kappas, deltas = [1.0, 1.3, 1.6], [0.0, 0.2, 0.4]
    for k, kap in enumerate(kappas):
        for j, dl in enumerate(deltas):
            bb = build_lcfs(r0=0.0, z0=0.0, a=1.0, kappa=kap, delta=dl, npts=200)
            ax.plot(bb[:, 0] + j * 3.0, bb[:, 1] + (1 - k) * 4.0,
                    "-", lw=0.9, color=plt.cm.viridis(0.12 + 0.35 * k))
    for j, dl in enumerate(deltas):
        ax.text(j * 3.0, -6.9, f"$\\delta={dl}$", ha="center", fontsize=6.5)
    for k, kap in enumerate(kappas):
        ax.text(-2.7, (1 - k) * 4.0, f"$\\kappa={kap}$", va="center", ha="center", fontsize=6.5)
    ax.set_xlim(-4.0, 7.4); ax.set_ylim(-7.6, 6.4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(c) what the shape parameters do", fontsize=8)

    fig.savefig(out / "fig_primer.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out/'fig_primer.pdf'}")


def fig_predicted_vs_actual(frozen, cache: dict[str, Path], out: Path) -> None:
    """Four predictors against the truth, and what the error metric misses."""
    plt = _style()

    case = int(np.argmin(np.abs(frozen["X"][:, 2] - 1.35)))
    true = frozen["psi"][case]
    lcfs = _lcfs_for(frozen["records"][case])
    vmin, vmax = float(np.nanmin(true)), float(np.nanmax(true))

    show = [
        ("best",         "the best run in the campaign"),
        ("unmasked",     "accurate inside, invents a field outside"),
        ("constant",     "right mask, almost no variation"),
        ("catastrophic", "right mask, five times worse than the mean map"),
    ]
    rows = 1 + len(show)
    fig, axes = plt.subplots(rows, 2, figsize=(4.6, 1.28 * rows),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.16,
                                          "left": 0.34, "right": 0.84})

    # Row 0: the truth, and where the plasma actually is.
    pc = _psi_panel(axes[0, 0], frozen["R"], frozen["Z"], true,
                    vmin=vmin, vmax=vmax, lcfs=lcfs)
    axes[0, 0].set_title("field", fontsize=7.5, pad=3)
    axes[0, 1].pcolormesh(frozen["R"], frozen["Z"], np.isfinite(true).astype(float),
                          cmap="Greys", vmin=0, vmax=1.7, shading="auto")
    axes[0, 1].plot(lcfs[:, 0], lcfs[:, 1], "k-", lw=0.9)
    axes[0, 1].set_aspect("equal"); axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])
    axes[0, 1].set_title("where $\\psi$ is defined", fontsize=7.5, pad=3)
    for s_ in axes[0, 1].spines.values():
        s_.set_linewidth(0.4)
    axes[0, 0].text(-0.08, 0.5, "ground truth", transform=axes[0, 0].transAxes,
                    ha="right", va="center", fontsize=7.5)

    for r, (label, blurb) in enumerate(show, start=1):
        psi = load_cached(cache[label])[case]
        cond, _run_id, _ = SELECTED[label]
        rl = _rel_l2_one(psi, true)
        _psi_panel(axes[r, 0], frozen["R"], frozen["Z"], psi,
                   vmin=vmin, vmax=vmax, lcfs=lcfs)

        diff = psi - np.nan_to_num(true, nan=0.0)
        diff = np.where(np.isfinite(true) | np.isfinite(psi), diff, np.nan)
        # Per-row scale: a scale shared with the catastrophic run would render
        # the other three uniformly blank. The peak is printed on each panel so
        # the rows stay comparable by number even though the colours are not.
        dmax = float(np.nanmax(np.abs(diff))) or 1.0
        _psi_panel(axes[r, 1], frozen["R"], frozen["Z"], diff, vmin=-dmax, vmax=dmax,
                   cmap="PuOr_r", contours=False, lcfs=lcfs)
        axes[r, 1].text(0.97, 0.03, f"$\\pm${dmax*1e3:.2g} mWb",
                        transform=axes[r, 1].transAxes, ha="right", va="bottom",
                        fontsize=5.6, color="0.25")
        axes[r, 0].text(-0.08, 0.60, f"{cond},  rel-$L_2$ {rl:.3f}",
                        transform=axes[r, 0].transAxes, ha="right", va="center",
                        fontsize=7)
        axes[r, 0].text(-0.08, 0.36, blurb, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=6.1, color="0.35",
                        wrap=True)

    axes[1, 0].set_title("prediction", fontsize=7.5, pad=3)
    axes[1, 1].set_title("prediction $-$ truth", fontsize=7.5, pad=3)

    cax = fig.add_axes([0.865, 0.16, 0.017, 0.26])
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label(r"$\psi$ [Wb]", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)
    cb.locator = plt.MaxNLocator(3); cb.update_ticks()

    fig.savefig(out / "fig_predicted_vs_actual.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out/'fig_predicted_vs_actual.pdf'}")


def _rel_l2_one(pred: np.ndarray, true: np.ndarray) -> float:
    m = np.isfinite(true)
    denom = float(np.linalg.norm(true[m]))
    num = float(np.linalg.norm(np.nan_to_num(pred[m]) - true[m]))
    return num / denom if denom else float("inf")


def fig_error_calibration(frozen, cache: dict[str, Path], out: Path) -> None:
    """What each value of the error metric actually looks like."""
    plt = _style()

    case = int(np.argmin(np.abs(frozen["X"][:, 2] - 1.35)))
    true = frozen["psi"][case]
    lcfs = _lcfs_for(frozen["records"][case])
    vmin, vmax = float(np.nanmin(true)), float(np.nanmax(true))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_map = np.nanmean(frozen["psi"], axis=0)

    panels = [("truth", true)]
    for label in ("best", "good", "typical_l3", "typical_l2"):
        panels.append((label, load_cached(cache[label])[case]))
    panels.append(("mean map", mean_map))
    panels.append(("constant", load_cached(cache["constant"])[case]))

    fig, axes = plt.subplots(1, len(panels), figsize=(1.02 * len(panels), 1.65),
                             gridspec_kw={"wspace": 0.08})
    for ax, (label, field) in zip(axes, panels):
        _psi_panel(ax, frozen["R"], frozen["Z"], field, vmin=vmin, vmax=vmax, lcfs=lcfs)
        if label == "truth":
            ax.set_title("truth", fontsize=7)
        else:
            ax.set_title(f"{_rel_l2_one(field, true):.3f}", fontsize=7)
        ax.set_xlabel("" if label == "truth" else
                      ("mean map" if label == "mean map" else SELECTED[label][0]),
                      fontsize=6, labelpad=2)
    axes[0].text(-0.12, 0.5, "rel-$L_2$:", transform=axes[0].transAxes,
                 rotation=90, va="center", ha="right", fontsize=6.5, color="0.35")
    fig.savefig(out / "fig_error_calibration.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out/'fig_error_calibration.pdf'}")


def fig_task_character(report: Path, out: Path) -> None:
    """Why the task is hard, and hard for the right reasons.

    Reads the characterisation sweep (pool n = 9993) that the paper currently
    reports only as appendix tables. A curve makes the model-family switch and
    the out-of-distribution collapse legible at a glance; a table does not.
    """
    plt = _style()
    d = json.loads(report.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.95), gridspec_kw={"wspace": 0.34})

    # (a) learning curve, with the winning model family marked
    ax = axes[0]
    ns = d["expA_learning_curve"]["n_grid"]
    best = d["expA_learning_curve"]["best_per_n"]   # keyed by str(N)
    ratios = [best[str(n)]["ratio"] for n in ns]
    fams = [best[str(n)]["model"] for n in ns]
    ax.plot(ns, ratios, "-", color="0.55", lw=0.9, zorder=1)
    seen = set()
    palette = {"poly_ridge": "#B8860B", "kernel_ridge": "#1F4E79"}
    for n, r, fam in zip(ns, ratios, fams):
        ax.scatter(n, r, s=16, color=palette.get(fam, "0.4"), zorder=3,
                   label=fam.replace("_", " ") if fam not in seen else None)
        seen.add(fam)
    ax.axvspan(150, 450, color="#cfe0f0", alpha=0.45, zorder=0)
    ax.text(260, 0.245, "the agents'\nbudget", fontsize=5.8, ha="center",
            va="bottom", color="#1F4E79")
    ax.set_xscale("log"); ax.set_xlabel("training solves $N$")
    ax.set_ylabel("error / mean-map baseline")
    ax.set_ylim(0.2, 0.68)
    ax.legend(frameon=False, fontsize=6, loc="upper right", borderpad=0.1)
    ax.set_title("(a) more data helps, and the\nbest model family changes", fontsize=7.5)

    # (b) where you sample, at matched budget
    ax = axes[1]
    cov = d["expB_coverage"]
    styles = {"space_filling": ("#1F4E79", "o", "space-filling"),
              "random": ("#B8860B", "s", "uniform random"),
              "clustered": ("#8c2d2d", "^", "clustered")}
    for strat, (col, mk, lab) in styles.items():
        pts = sorted([(c["n_train"], c["ratio"]) for c in cov if c["strategy"] == strat])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", color=col,
                ms=3.4, lw=0.9, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("training solves $N$")
    ax.set_ylabel("error / mean-map baseline")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("(b) at matched budget, where\nyou sample matters", fontsize=7.5)

    # (c) out-of-distribution collapse
    ax = axes[2]
    shift = {s["shift"]: s for s in d["expD_shift"]}
    order = ["train_low\u2192test_low", "train_high\u2192test_high",
             "train_low\u2192test_high", "train_high\u2192test_low"]
    labels = ["low $\\rightarrow$ low", "high $\\rightarrow$ high",
              "low $\\rightarrow$ high", "high $\\rightarrow$ low"]
    vals = [shift[o]["ratio"] for o in order]
    cols = ["#1F4E79", "#1F4E79", "#8c2d2d", "#8c2d2d"]
    ax.barh(range(4), vals, color=cols, height=0.62)
    ax.axvline(1.0, color="0.35", lw=0.8, ls=(0, (4, 2)))
    ax.text(0.98, 0.15, "no better than\nthe mean map", fontsize=5.8, color="0.35",
            va="bottom", ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("error / mean-map baseline"); ax.set_xlim(0, 1.12)
    ax.set_title("(c) outside its training region the\nsurrogate collapses", fontsize=7.5)

    fig.savefig(out / "fig_task_character.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out/'fig_task_character.pdf'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="all", choices=("cache", "render", "all"))
    ap.add_argument("--tag", default="matrix-v4-20260919")
    ap.add_argument("--out", default="docs/paper/neurips_benchmark/figures")
    ap.add_argument("--cache-dir", default=None,
                    help="default: experiments/<tag>/prediction_cache")
    ap.add_argument("--force", action="store_true", help="re-run cached predictions")
    args = ap.parse_args()

    tag_dir = REPO_ROOT / "experiments" / args.tag
    cache_dir = Path(args.cache_dir) if args.cache_dir else tag_dir / "prediction_cache"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.stage in ("cache", "all"):
        cache_predictions(tag_dir, cache_dir, force=args.force)
    if args.stage in ("render", "all"):
        from matrix_report import load_frozen
        frozen = load_frozen()
        cache = {label: cache_dir / f"{c}__{r}.npz"
                 for label, (c, r, _) in SELECTED.items()}
        missing = [k for k, p in cache.items() if not p.exists()]
        if missing:
            print(f"missing cached predictions: {missing} — run the cache stage",
                  file=sys.stderr)
            return 1
        fig_primer(frozen, out)
        report = REPO_ROOT / "benchmarks" / "assets" / "scaling_report.json"
        if report.is_file():
            fig_task_character(report, out)
        else:
            print(f"[skip] {report} absent — task-character figure not drawn",
                  file=sys.stderr)
        fig_predicted_vs_actual(frozen, cache, out)
        fig_error_calibration(frozen, cache, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
