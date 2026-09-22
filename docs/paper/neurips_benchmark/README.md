# `neurips_benchmark/` — Overleaf-ready LaTeX source

*"Paying for Your Own Data: A Closed-Loop Benchmark for Scientific ML Agents"*,
typeset for the **NeurIPS 2026 Evaluations & Datasets track** (the track
formerly called Datasets & Benchmarks). The draft is complete: every number is
measured, every open slot is closed, and the Paper Checklist is answered.

## Import into Overleaf

Upload `neurips_benchmark.zip` (sibling of this folder) via **New Project →
Upload Project**, or drag the folder in. Then set:

- **Compiler**: pdfLaTeX
- **Main document**: `main.tex`

Nothing else to configure.

## Current state

```
./check_compliance.sh
  build integrity ........ PASS (no undefined refs, no errors, fonts embedded,
                                 no Type 3, zero overfull boxes)
  unresolved content ..... PASS (0 FILL slots, 0 \tbd cells)
  page limit ............. PASS (countable body ends on page 9 of 9)
  required components .... PASS (checklist present, official style file,
                                 no unverified bibliography entries)
```

30 pages total: 9 of body, then acknowledgments, references, appendices A–P and
the Paper Checklist, none of which count against the limit.

## Two things to check before you submit

1. **The style file came from a public mirror of the NeurIPS 2026 author kit**,
   not from `neurips.cc` directly (their server blocks scripted downloads). It
   self-identifies as `neurips_2026.sty [2026-01-29]` by Roman Garnett and
   declares the expected track options, so it is almost certainly the real
   thing — but diff it against the official kit before submitting, since a
   wrong text block is a desk-reject. The same applies to
   `sections/12_checklist.tex`.
2. **Authorship.** The draft is a named preprint. Confirm the co-author line
   before posting, and drop `preprint` from the style options for double-blind
   review (`\usepackage[eandd]{neurips_2026}`), which blanks the author block
   and turns line numbers on. Appendix C then needs its repository URL
   anonymised; `check_compliance.sh` greps for the obvious leaks.

The track option is **`eandd`**, not `datasets`: for 2026 the Datasets &
Benchmarks track became the Evaluations & Datasets track, and `datasets` is not
a declared option of the real style file.

## Layout

```
main.tex            preamble, title, abstract, \input order
macros.tex          notation shortcuts (\Ltwo, \hn, \code, ...)
neurips_2026.sty    venue style — see the caveat above
references.bib      27 entries, every one web-verified; the log is in
                    ../references_verification.md
sections/           01 intro · 02 related · 03 benchmark · 05 design
                    06 results · 07 verification · 08 agent logic
                    09 limitations · 10 conclusion · 11 appendix
                    12 checklist (verbatim from the author kit, answered)
figures/            fig_levels, fig_verification, fig_cost — generated
tables/             task text, L2/L3 diff, solution shape, glossary — generated
```

## Regenerating the figures and appendix tables

They are not hand-written; they are emitted from the campaign artifacts so they
cannot drift from the numbers they document:

```bash
python tools/render_paper_figures.py --tag matrix-v4-20260919 \
       --out docs/paper/neurips_benchmark/figures
python tools/render_paper_tables.py  --tag matrix-v4-20260919 \
       --out docs/paper/neurips_benchmark/tables
```

Every number in the prose is traceable through `../results_v4.md`, which names
the file and the line of code behind each one.

## Retargeting another venue

| Target | Change |
|---|---|
| NeurIPS E&D (current) | `\usepackage[preprint,eandd]{neurips_2026}` |
| Anonymous for review | drop `preprint` |
| Camera ready | `preprint` → `final` |
| ICML / ICLR | swap the `\usepackage` line; `sections/` are style-agnostic |
| *Mach. Learn.: Sci. Technol.* | swap to `iopart.cls`; §3 and §7 carry the physics-ML weight. No page limit there, so the appendix material can come forward |

## Build locally

```bash
latexmk -pdf main.tex     # build
./check_compliance.sh     # pre-submission check (--strict to fail on any FAIL)
latexmk -c                # clean aux, keep the PDF
```
