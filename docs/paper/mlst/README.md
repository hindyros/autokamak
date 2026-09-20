# `mlst/` — the full-length paper

*"Paying for Your Own Data: A Closed-Loop Benchmark for Scientific
Machine-Learning Agents"*, typeset for **Machine Learning: Science and
Technology** (IOP).

This is the current version of the paper. It exists because the 9-page
conference draft in `../neurips_benchmark/` was unreadable to anyone without
context: to fit the limit, the benchmark-loop diagram had been pushed into the
appendix, the Grad–Shafranov equation arrived on page one with no explanation of
what a tokamak or ψ is, and there was not a single physics figure. With no page
limit, all of that comes back.

## Build

```bash
latexmk -pdf main.tex
```

41 pages: 26 of body, then the data availability statement, six appendices and
the references. Zero errors, zero overfull boxes, zero undefined references.

One benign warning remains — `Command \@xhline has changed`, which is booktabs
noticing that `iopart.cls` redefined it. The rules render correctly.

## What is different from the conference draft

- **A new §2**, "The problem, for a reader who has never seen a tokamak":
  what a tokamak is, what ψ means, why ψ is undefined outside the boundary and
  why that convention turns out to matter enormously, what the five shape
  parameters do, and why anyone wants a fast surrogate for this solver.
- **Four new figures**, all generated from campaign artifacts by
  `tools/render_physics_figures.py`: the physics primer, an error-metric
  calibration strip, the predicted-vs-actual comparison (Figure 7, the one that
  makes the verification argument visible), and the task-difficulty analysis
  that used to be three appendix tables.
- **The benchmark-loop diagram is back on page 3**, where it belongs.
- **A vocabulary table** in §4.1 and definitions before first use throughout.
- **A worked example** (§4.6): one run end to end, from the task text it
  received to what the diagnostics said about it.
- **A data availability statement**, which IOP requires.
- Compressed run-on paragraphs broken back into prose.

## Class files

`iopart.cls`, `iopart12.clo`, `iopams.sty` and `setstack.sty` are IOP
Publishing's own, redistributed here unmodified under the LPPL. Two known
interactions with our preamble are handled in `main.tex` and commented there:
`iopart` defines `equation*` before `amsmath` can, and it does not define
`\newblock`, which the natbib bibliography styles emit.

MLST accepts any reasonable format for initial submission — IOP does the
typesetting in production — so this being IOP-styled is a convenience, not a
requirement.

## Regenerating figures and appendix tables

Neither is hand-written; both are emitted from the campaign so they cannot
drift from the numbers they document:

```bash
python tools/render_paper_figures.py   --tag matrix-v4-20260919 --out docs/paper/mlst/figures
python tools/render_physics_figures.py --tag matrix-v4-20260919 --out docs/paper/mlst/figures
python tools/render_paper_tables.py    --tag matrix-v4-20260919 --out docs/paper/mlst/tables
```

The physics figures need agent predictions on the frozen set, which the
benchmark never persisted. The first run of `render_physics_figures.py` re-runs
eight agents' `predict.py` offline and caches the fields under
`experiments/<tag>/prediction_cache/`; later runs read the cache.

## Provenance of the numbers

Every figure in the prose is traceable through `../results_v4.md`, which names
the artifact and the line of code behind each one. Every citation was checked by
web search before it entered `references.bib`; the log is
`../references_verification.md`.
