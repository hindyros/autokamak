# Papers

## Current paper: `mlst/` — and `mlst.zip` for Overleaf

The full-length version, targeting **Machine Learning: Science and Technology**
(IOP). This is the one to read and the one to submit. It exists because the
9-page conference draft was opaque to anyone without context; with no page
limit the explanation and the physics figures go back in. 41 pages, builds
clean. See `mlst/README.md`.

## Frozen conference draft: `neurips_benchmark/`

The 9-page NeurIPS Evaluations & Datasets version, complete and compliant as of
commit `f6b0df8`: every number measured, the Paper Checklist answered,
`check_compliance.sh` passing. **It is no longer maintained** — `mlst/` is. It
is kept because the 9-page form is genuinely hard to reconstruct, and because
the conference route may still be wanted. If you revive it, note that the
physics figures added later are not wired into it.

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
