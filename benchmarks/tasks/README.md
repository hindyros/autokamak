# Task specifications

Ten task files live here and only two of them are current. This note says
which, because the difference is not guessable from the filenames.

## Current — used for the published campaign

| File | Level | Prompt version |
|---|---|---|
| `L2_mini_v3.yaml` | L2, may import the platform library | 3 |
| `L3_mini_v3.yaml` | L3, from scratch, import audited | 3 |

These two are byte-identical except for ten passages, all concerning access
level. The budget is at most 150 initial solves plus up to three adaptive
rounds of exactly 100; the timeout is 5400 s; `feedback_rounds` is 1.

`smoke.yaml` is the cheap end-to-end check to run against a new harness before
spending money. Always smoke first.

## Superseded — kept for provenance, do not run

`L2_library.yaml`, `L3_from_scratch.yaml` (v1, unversioned), their `_v2`
variants, and `L2_mini.yaml` / `L3_mini.yaml` / `L2_mini_v2.yaml` /
`L3_mini_v2.yaml`.

Why they were replaced, in order:

- **v1 → v2** (2026-08-10) added two process gates after a campaign in which an
  agent stored all-NaN datasets, counted them as successes, and shipped a
  `predict.py` it had never executed: a *storage-validation gate* (a solve
  counts only once its stored file has been read back and confirmed finite) and
  a *deliverable self-test* (run every documented command in a fresh process
  before declaring done).
- **v2 → v3** (2026-08-18) fixed budget numbers left over from the full-size
  tasks. v2 stated both 500-solve and 100-solve rounds in different places, so
  an agent could legitimately read a 17× spread in the solver budget it was
  allowed. v3 says 100 × 3 rounds, once.

**Do not pool results across prompt versions.** `tools/aggregate_matrix.py`
refuses to, and that refusal is deliberate: the text is what the agent was
asked to do, so changing it changes the experiment.
