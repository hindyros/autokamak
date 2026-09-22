# Published results

The result tables behind the paper, committed so every number in it can be
checked without re-running a $200 campaign.

`matrix-v4-20260919/` is the published campaign. It is the small, verifiable
core of the full results bundle:

| file | what it is |
|---|---|
| `aggregate.csv` | one row per cell: pass rates with intervals, error medians with bootstrap CIs, validity and honesty counts, cost |
| `cost_report.csv` | one row per run: tokens, dollars, wall time, solver calls, measured vs derived |
| `methodology.csv` | what each agent actually built, read from its code rather than its claims |
| `methodology_rounds.csv` | per-round campaign behaviour |
| `solution_shape.csv` | the shape of each delivered solution |
| `judge_scores.csv` | blind LLM-judge rubric scores |
| `DATA_DICTIONARY.md` | every column in every table, defined |
| `GLOSSARY.md` | every term and code used in the tables and the report |
| `*_stdout.txt` | the exact console output the analysis produced |

**Not committed here:** the browsable HTML report (~10 MB), the 59 per-run
records with transcripts, and the agent workspaces (~10 GB of raw campaign
output). Rebuild the full 17 MB bundle from a campaign directory with:

```bash
python tools/make_results_bundle.py --tag matrix-v4-20260919
```

Regenerate the campaign itself with the steps in the README's *Reproducing
the paper* section.
