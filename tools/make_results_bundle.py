#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Package a campaign's results as a self-contained folder someone else can read.

The campaign directory is 10 GB, nearly all of it agent workspaces and raw
event streams. What another researcher actually needs -- the aggregated tables,
the statistics, the browsable report, the per-run records and transcripts --
is about 12 MB, and none of it is documented anywhere. This builds that folder,
with a README and a data dictionary, so the results can be handed to someone
who was not there.

    python tools/make_results_bundle.py --tag matrix-v4-20260919
    python tools/make_results_bundle.py --tag <tag> --out /somewhere/else

Deliberately excluded: agent workspaces (3.2 GB of datasets and model
pickles) and the harness event streams (6.9 GB). Both are reachable in the
repository if anyone needs them; neither belongs in a folder meant to be
opened and read.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundle_docs import write_data_dictionary, write_readme  # noqa: E402

# Top-level analysis outputs, in the order a reader should meet them.
TAG_FILES = [
    ("index.html",              "the browsable report -- start here"),
    ("aggregate.csv",           "one row per experimental cell; the paper's main table"),
    ("methodology.csv",         "one row per run: the chain of methods it chose"),
    ("methodology_rounds.csv",  "one row per adaptive round: the decision logic"),
    ("solution_shape.csv",      "12 design questions x 8 cells, from static analysis"),
    ("cost_report.csv",         "one row per run: dollars, tokens, wall time, solves"),
    ("judge_scores.csv",        "blind code-review rubric scores"),
    ("aggregate_stdout.txt",    "the statistics: pooled contrast and all three tests"),
    ("judge_score_stdout.txt",  "console output of the code review"),
]

SUPPORTING = [
    ("library-baseline-matched/library_baseline.json",
     "the zero-LLM pipeline at a matched 450-solve budget, three seeds"),
    ("library-baseline/library_baseline.json",
     "the same pipeline with unlimited data (see the caveat in the README)"),
    ("control-arm-v2-20260919/control_arm_results.json",
     "forced adaptive vs forced space-filling, four paired replicates"),
]


def copy_tag_files(tag_dir: Path, dest: Path) -> list[str]:
    missing = []
    for name, _ in TAG_FILES:
        src = tag_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)
        else:
            missing.append(name)
    return missing


def copy_supporting(exp_dir: Path, dest: Path) -> None:
    out = dest / "supporting_arms"
    out.mkdir(exist_ok=True)
    for rel, _ in SUPPORTING:
        src = exp_dir / rel
        if src.is_file():
            shutil.copy2(src, out / (rel.split("/")[0] + ".json"))
    cfg = exp_dir / "control-arm-v2-20260919" / "configs"
    if cfg.is_dir():
        shutil.copytree(cfg, out / "control_arm_configs", dirs_exist_ok=True)


def copy_per_run(tag_dir: Path, dest: Path, *, traces: bool) -> tuple[int, int]:
    """One directory per run holding its record and, optionally, its transcript.

    Never touches `workspace/` or the `*_events.jsonl` streams, and never
    follows the OpenFUSIONToolkit symlink several workspaces contain.
    """
    out = dest / "runs"
    out.mkdir(exist_ok=True)
    n_res = n_tr = 0
    for result in sorted(tag_dir.glob("*/*/result.json")):
        run = out / result.parent.parent.name / result.parent.name
        run.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result, run / "result.json")
        (run / "result.json").chmod(0o644)
        n_res += 1
        trace = result.parent / "trace.json"
        if traces and trace.is_file():
            shutil.copy2(trace, run / "trace.json")
            (run / "trace.json").chmod(0o644)   # originals are 0600
            n_tr += 1
        judge = result.parent / "eval" / "judge.json"
        if judge.is_file():
            shutil.copy2(judge, run / "judge.json")
            (run / "judge.json").chmod(0o644)
    return n_res, n_tr


def copy_paper(dest: Path) -> str | None:
    for rel in ("docs/paper/mlst/main.pdf", "docs/paper/neurips_benchmark/main.pdf"):
        src = REPO_ROOT / rel
        if src.is_file():
            out = dest / "paper"
            out.mkdir(exist_ok=True)
            shutil.copy2(src, out / "paper.pdf")
            figs = src.parent / "figures"
            if figs.is_dir():
                shutil.copytree(figs, out / "figures", dirs_exist_ok=True)
            return rel
    return None


def write_glossary(dest: Path) -> int:
    """The cell vocabulary, straight from the module that defines it."""
    from autotokamak.bench import glossary
    groups = glossary.all_terms()
    lines = ["# Glossary of cell values", "",
             "Every canonical token that appears in `methodology.csv`,",
             "`solution_shape.csv` and the HTML report. Generated from",
             "`src/autotokamak/bench/glossary.py`, so it cannot drift from the code",
             "that emits these values.", ""]
    n = 0
    for group, terms in groups.items():
        lines += [f"## {group}", ""]
        for token, definition in terms.items():
            lines.append(f"**`{token}`** — {' '.join(definition.split())}")
            lines.append("")
            n += 1
    (dest / "GLOSSARY.md").write_text("\n".join(lines))
    return n


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="matrix-v4-20260919")
    ap.add_argument("--out", default=None,
                    help="default: ~/Desktop/autotokamak-results-<tag>")
    ap.add_argument("--no-traces", action="store_true",
                    help="omit the agent transcripts (they are included by default)")
    ap.add_argument("--experiments-dir", default=None)
    args = ap.parse_args()

    exp_dir = Path(args.experiments_dir) if args.experiments_dir else REPO_ROOT / "experiments"
    tag_dir = exp_dir / args.tag
    if not tag_dir.is_dir():
        print(f"no such campaign: {tag_dir}", file=sys.stderr)
        return 1

    dest = Path(args.out).expanduser() if args.out else \
        Path.home() / "Desktop" / f"autotokamak-results-{args.tag}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    missing = copy_tag_files(tag_dir, dest)
    copy_supporting(exp_dir, dest)
    n_res, n_tr = copy_per_run(tag_dir, dest, traces=not args.no_traces)
    paper = copy_paper(dest)
    n_terms = write_glossary(dest)
    write_readme(dest, args.tag, n_res, n_tr, paper, missing)
    write_data_dictionary(dest, tag_dir)

    size = dir_size(dest)
    print(f"Wrote {dest}")
    print(f"  {n_res} run records, {n_tr} transcripts, {n_terms} glossary terms")
    if missing:
        print(f"  missing from the campaign: {', '.join(missing)}", file=sys.stderr)
    print(f"  total {size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
