# Publishing this repository

> This file is for the maintainer and can be deleted from the public copy.

The plan is to publish a **fresh repository with a squashed initial commit**,
keeping this one private as the lab notebook. The reason is concrete: ~150
absolute paths of the form `/Users/hindy/Desktop/Deep Deka/autokamak` were
tracked across 43 files for months. The working tree is clean now, but
`git log -p` still shows them, along with a username, a desktop layout, an
internal project name, and thirteen work-in-progress branches. A squashed
initial commit removes all of that from public view at no cost to anyone.

Nothing below has been run. Run it yourself — publishing is not a step to
delegate.

## 1. Pre-flight

```bash
# Everything must be clean and on main.
git switch main && git merge --no-ff feat/task-prompt-v2
ruff check src tools tests
pytest tests/ -q

# No developer paths anywhere in the tracked tree.
git grep -nI "/Users/\|/home/[a-z]" -- . ':!*.ipynb' && echo "FOUND -- stop" || echo "clean"

# No key-shaped strings.
git grep -nIE "sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}" -- . \
  && echo "FOUND -- stop" || echo "clean"

# The measurement path is undisturbed: re-aggregate and diff.
python tools/aggregate_matrix.py --tag matrix-v4-20260919
git diff --stat experiments/ 2>/dev/null   # gitignored, so expect nothing
```

## 2. Build the public tree

```bash
PUB=~/autotokamak-public
rm -rf "$PUB" && mkdir -p "$PUB"

# Copy the tracked tree only. `git archive` cannot pick up anything ignored,
# untracked, or lurking in history, which is exactly the property we want.
git archive HEAD | tar -x -C "$PUB"

rm -f "$PUB/PUBLISHING.md"      # this file
cd "$PUB"
```

Then check the result on its own terms, not on this repository's:

```bash
grep -rn "Deep Deka" . ; echo "--- expect nothing above ---"
grep -rln "/Users/" . ; echo "--- expect nothing above ---"
du -sh .                        # expect tens of MB, not hundreds
ls                              # README, LICENSE, CITATION.cff, CONTRIBUTING
```

## 3. Prove it works for someone who is not you

The point of the exercise. Use a clean interpreter, not the existing venv:

```bash
cd "$PUB"
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,dev,harnesses]"
python -c "from autotokamak.bench import compute_diagnostics, validate_deliverables; print('API ok')"
ruff check src tools tests
pytest tests/ -q                 # 9 will skip: the gitignored dataset
```

If any of that fails, the public copy is broken for everyone else even though
it works here — which is the whole reason for doing it in a fresh directory.

## 4. Publish

```bash
cd "$PUB"
git init -b main
git add -A
git commit -m "autotokamak: a closed-loop benchmark for scientific ML agents

A benchmark in which a coding agent must generate its own training data from
an expensive Grad-Shafranov solver under a fixed budget, then ship a predictor
scored independently on a frozen set it never sees. Includes the verification
layer, the deterministic method-chain analysis, and the tooling to reproduce
the published campaign."

gh repo create autotokamak --public --source=. --remote=origin \
   --description "A closed-loop benchmark for scientific ML agents: the agent has to buy its own data"
git push -u origin main
```

**Name it `autotokamak`**, matching the package. The current remote is
`autokamak` (no `t`), and that mismatch already confuses the metadata. If you
keep the old name instead, change `[project.urls]` in `pyproject.toml`,
Appendix B of the paper, and the data availability statement to match — the
paper cites the URL, so the two must agree before submission.

## 5. Afterwards

- Add a topic list on GitHub: `benchmark`, `llm-agents`, `fusion`,
  `surrogate-models`, `scientific-machine-learning`.
- Archive a release to Zenodo for a DOI, and put that DOI in `CITATION.cff`
  and in the paper's data availability statement.
- Confirm the Actions run is green on the fresh repository; CI has never
  executed anywhere else.
