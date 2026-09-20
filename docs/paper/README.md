# Papers

## Submission source: `neurips_benchmark/`

The LaTeX build of the benchmark paper, targeting the **NeurIPS 2026
Evaluations & Datasets track**, plus `neurips_benchmark.zip` for direct
Overleaf import (New Project → Upload Project; compiler pdfLaTeX, main document
`main.tex`). **The draft is complete**: every number is measured from
`experiments/matrix-v4-20260919`, no `\FILL` slots remain, the Paper Checklist
is answered, and `check_compliance.sh` passes every check. 9 pages of body, 30
total. See `neurips_benchmark/README.md` for the two things to verify before
submitting.

## `results_v4.md` — the numbers ledger

Every figure the paper quotes, with the artifact and the line of code it came
from. If a number is not in this file it does not belong in the paper; check
the .tex against it rather than against a draft.

## `references_verification.md` — the citation log

Each bibliography entry, the query that checked it, the URL that confirmed it,
and what was corrected. Four entries previously flagged `% VERIFY` are
resolved; six new entries were added, all verified. Two candidate citations
were deliberately dropped because their metadata could not be confirmed.

## Prose original: `benchmark_paper.md`

The empirical paper. Its subject is the **benchmark** — can a coding agent
execute a closed-loop scientific-ML pipeline where it must pay for its own
data out of a fixed solver budget, and how much does pre-written library
code substitute for capability? Results come from the replicated prompt-v3
campaign (`experiments/matrix-v3-n10-20260917`, 8 cells × n=10, all on one
model) plus the verification diagnostics in `bench/diagnostics.py`.

## `system_description/` — the three older drafts

Three overlapping write-ups of the *system*. All are finished as system and
mathematics descriptions and all contain **zero empirical results**: no
Results section, no results table, no figure of measured performance, and no
mention of the harness × level matrix. They are not competing candidates for
submission; they are the source material for the benchmark paper's
"reference solution" appendix and its physics/method background.

| Draft | Files | What it is |
|---|---|---|
| System paper | `autotokamak_system_paper.{md,tex,pdf}` | Full-length system description, most complete of the three. |
| Summary | `autotokamak_summary.{tex,pdf}` | Two-page plain-language summary. |
| "autokamak" draft | `autokamak.{tex,pdf}` | Mathematical system description (note the different spelling). |
