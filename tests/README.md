# Tests

```bash
pytest tests/ -q          # the default: everything that runs offline and fast
pytest tests/ -q -m slow  # adds the tests that need a real Grad-Shafranov solve
```

No test touches the network or needs an API key. The language-model paths are
all mocked.

## Two markers, and why the counts move

**`slow`** — needs a real solve through the Open FUSION Toolkit. Deselected by
default via `addopts` in `pyproject.toml`. Two tests.

**`needs_dataset`** — needs `examples/dataset_generation/outputs/dataset.h5`,
a Phase-1 dataset that is gitignored because it is large. Nine end-to-end
tests, including the whole of `test_meta_loop_e2e_mock.py`, which is the
largest test file in the repository.

That second one is worth knowing about. On a fresh clone those nine tests skip,
and a green wall of dots hides the fact that the heaviest integration coverage
did not run. To get them:

```bash
python -m autotokamak.pipelines phase1 --n-samples 200   # a few minutes of solves
pytest tests/ -q -m needs_dataset                        # now they run
```

## What is and is not covered

Well covered: the deliverable contract and its gates, the physical-validity
diagnostics, the methodology and solution-shape extractors, the L0 scripted
policies, active-learning acquisition, the DSPy pickers, and the harness
interface itself (`test_harness_contract.py` instantiates all six adapters).

Thin: `tools/`. Roughly 7,000 lines of analysis scripts with almost no direct
tests, including `aggregate_matrix.py`, which produces the paper's main table.
It is exercised indirectly — the campaign is re-aggregated and compared
byte-for-byte after any refactor — but that is a convention, not a test.
