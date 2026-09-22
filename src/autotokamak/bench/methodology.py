# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Extract WHAT each agent did and WHY, as a comparable record.

Why this exists
---------------
Every other instrument in ``bench`` scores the OUTCOME: did the deliverable
contract hold, what is the relative-L2 against the frozen test set, is the
field physically valid. None of them can distinguish two runs that land on
the same error by entirely different reasoning — a run that picked its 100
adaptive points by ensemble disagreement and stopped because validation
error crossed the task's threshold, versus one that relabelled uniform
random sampling as "adaptive" and stopped because it ran out of rounds.
That difference is the research question of this repo, and until now it
survived only as prose in a README.

Two things are extracted, mirroring the two questions asked of a method:

1. **The chain of methods** (``method_chain`` / ``chain_signature``) — the
   final pipeline the agent settled on, canonicalised into one vocabulary:
   initial design -> representation -> model family -> HPO -> acquisition ->
   stopping rule. This is the "what".
2. **The per-iteration decision logic** (``iterations``) — for each adaptive
   round: the criterion the agent stated, what it had measured when it chose
   (validation error vs baseline), and what it decided next. This is the
   "why", round by round.

Design constraints, all deliberate
----------------------------------
* **Zero LLM, zero execution.** Deterministic over artifacts the agent
  already had to write (the v3 task mandates an acquisition log, a
  ``sampling_strategy`` string and ``report.json``), so the whole archived
  corpus can be extracted at no cost and re-extracted identically.
* **Measurement, never a gate.** Like ``bench.diagnostics``, nothing here
  touches ``contract.passed``. Adding a stage to the vocabulary changes no
  archived score.
* **One vocabulary across access levels.** The canonical acquisition terms
  are anchored on the L0/L1 typed action space
  (``agent.orchestrator.schema.AcquisitionStrategy``), so the scripted
  policy, the L1 typed picker and a from-scratch L3 agent are describable in
  the same words and land in the same table.
* **Unknown stays unknown.** Every field is Optional and nothing raises; a
  missing log is a finding, not a crash. ``sources`` records which file
  supplied each stage so a reader can audit any cell back to its evidence.

What it cannot do: it reads stated criteria, not implemented ones. An agent
that writes "ensemble disagreement" in its log and samples uniformly is
recorded as *claiming* that criterion — ``criterion_stated_vs_random`` and
the code-signal cross-check narrow this, and ``tools/judge_code.py`` remains
the instrument for the qualitative call.
"""
from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Bounds. Acquisition logs are agent-authored and occasionally enormous;
# extraction must stay cheap enough to run over a whole campaign.
MAX_LOG_BYTES = 32 * 1024 * 1024
MAX_LOG_RECORDS = 200_000
MAX_CODE_FILES = 200
MAX_CODE_BYTES = 400_000
MAX_CRITERION_CHARS = 300

EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules",
                ".ruff_cache", ".pytest_cache", ".ipynb_checkpoints",
                "OpenFUSIONToolkit", "ursa_metrics"}

# The task's own stopping rule: val_error <= 0.30 * baseline_error.
STOP_THRESHOLD_RATIO = 0.30


# ---------------------------------------------------------------------------
# Canonical vocabulary
# ---------------------------------------------------------------------------
# Patterns are matched case-insensitively against agent prose (log criteria,
# report.json sampling_strategy, README) and against agent code. Ordering
# within a stage does not matter; every match is recorded, because real
# pipelines combine (e.g. "distance-to-training + MC-dropout variance").

INITIAL_DESIGN_PATTERNS: dict[str, str] = {
    "lhs": r"latin[\s_-]?hypercube|\blhs\b|LatinHypercube|pydoe",
    "sobol": r"\bsobol\b",
    "halton": r"\bhalton\b",
    "maximin": r"maximin|max[\s_-]?min\s+distance|poisson[\s_-]?disk",
    "grid": r"full[\s_-]?factorial|factorial design|grid design",
    "uniform_random": r"uniform(?:ly)?[\s_-]?random|random uniform|rng\.uniform|np\.random\.uniform",
}

REPRESENTATION_PATTERNS: dict[str, str] = {
    "pca": r"\bpca\b|principal[\s_-]?component|TruncatedSVD",
    "pod_svd": r"\bpod\b|proper orthogonal|\bsvd\b|randomized_svd",
    "autoencoder": r"auto[\s_-]?encoder|latent (?:space|code)|encoder.{0,12}decoder",
    "spline_basis": r"spline|RBFInterpolator|radial basis",
    "per_pixel": r"per[\s_-]?pixel|pixel[\s_-]?wise|per[\s_-]?grid[\s_-]?point",
}

MODEL_PATTERNS: dict[str, str] = {
    "gp": r"gaussian[\s_-]?process|GaussianProcessRegressor|gpytorch|\bkriging\b|\bgpr\b",
    "kernel_ridge": r"KernelRidge|kernel[\s_-]?ridge",
    "poly_ridge": r"PolynomialFeatures|polynomial regression",
    "ridge_linear": r"\bRidge\b|LinearRegression|\blasso\b|ElasticNet|linear regression",
    "random_forest": r"RandomForest|ExtraTrees",
    "gradient_boosting": r"GradientBoosting|HistGradientBoosting|\bxgboost\b|\blightgbm\b|\bcatboost\b",
    "svr": r"\bSVR\b|support vector regress",
    "knn": r"KNeighbors|nearest[\s_-]?neighbou?r",
    "mlp_sklearn": r"MLPRegressor",
    "mlp_torch": r"torch\.nn|nn\.Linear|nn\.Module|\bpytorch\b",
    "cnn_decoder": r"nn\.Conv(?:2d|Transpose2d)|ConvTranspose|u[\s_-]?net|deconv",
    "rbf_interpolant": r"RBFInterpolator|\bRbf\b",
}

HPO_PATTERNS: dict[str, str] = {
    "optuna": r"\boptuna\b",
    "grid_search": r"GridSearchCV|grid[\s_-]?search",
    "random_search": r"RandomizedSearchCV|random[\s_-]?search",
    "cv_select": r"cross_val_score|KFold|cross[\s_-]?validat",
    "manual_sweep": r"manual (?:sweep|tuning)|hand[\s_-]?tuned|hard[\s_-]?coded hyper",
    "early_stopping": r"early[\s_-]?stopping|\bpatience\b",
}

ENSEMBLE_PATTERNS: dict[str, str] = {
    "deep_ensemble": r"\bensemble\b|n_models|committee",
    "mc_dropout": r"mc[\s_-]?dropout|monte[\s_-]?carlo dropout",
    "bagging": r"\bbagging\b|bootstrap resampl",
}

# Acquisition vocabulary. The first three names are the L0/L1 typed
# strategies verbatim (``AcquisitionStrategy``); the rest are terms this
# corpus produced that the typed space has no word for.
ACQUISITION_PATTERNS: dict[str, str] = {
    "uncertainty": r"uncertain|posterior (?:variance|std)|predictive (?:variance|std)"
                   r"|epistemic|\bucb\b|expected improvement|\bei\b\b",
    "uncertainty_ensemble": r"ensemble[\w\s_-]{0,24}(?:variance|disagreement|std|spread)"
                            r"|disagree|query[\s_-]?by[\s_-]?committee|mc[\s_-]?dropout"
                            r"|monte[\s_-]?carlo[\s_-]?dropout|deep[\s_-]?ensemble",
    "uncertainty_gp": r"gp[\s_-]?(?:variance|std)"
                      r"|gaussian[\s_-]?process[\s_-]?(?:variance|posterior)"
                      r"|kriging[\s_-]?variance",
    "residual_ucb": r"residual|model error|error[\s_-]?driven|high[\s_-]?error"
                    r"|worst[\s_-]?case error|loo[\s_-]?cv error|out[\s_-]?of[\s_-]?fold error",
    "space_filling": r"space[\s_-]?filling|coverage|distance[\s_-]?to[\s_-]?training"
                     r"|maximin|farthest[\s_-]?point|diversity|novelty|k[\s_-]?means",
    "gradient_sensitivity": r"gradient|jacobian|sensitivit|curvature|steep",
    "feasibility": r"feasib|unconverged|failure[\s_-]?(?:rate|region)|success[\s_-]?rate",
    "random": r"\brandom\b|\buniform\b|\bi\.?i\.?d\.?\b",
}

STOP_RULE_PATTERNS: dict[str, str] = {
    "val_threshold_70pct": r"0\.3\s*\*|30\s*%|70\s*%|0\.30\s*(?:\*|x)|70 percent"
                           r"|reduc\w+ by at least 70",
    "round_cap": r"round cap|max(?:imum)? rounds|cap of 3|3[\s_-]?round|all 3 rounds",
    "plateau": r"plateau|no further improvement|diminishing return|converged",
    "budget": r"solve budget|budget exhaust|out of budget",
}

STAGE_PATTERNS: dict[str, dict[str, str]] = {
    "initial_design": INITIAL_DESIGN_PATTERNS,
    "representation": REPRESENTATION_PATTERNS,
    "model_family": MODEL_PATTERNS,
    "hpo": HPO_PATTERNS,
    "ensembling": ENSEMBLE_PATTERNS,
    "acquisition": ACQUISITION_PATTERNS,
    "stopping_rule": STOP_RULE_PATTERNS,
}

_COMPILED = {stage: {name: re.compile(pat, re.I) for name, pat in pats.items()}
             for stage, pats in STAGE_PATTERNS.items()}

# Chain order = the order the pipeline actually executes in, so the
# signature reads as a pipeline rather than as an alphabetised bag.
CHAIN_ORDER = ("initial_design", "representation", "model_family", "ensembling",
               "hpo", "acquisition", "stopping_rule")


# Every canonical term this module can emit, defined. Rendered as hover text
# wherever a code appears in a report: a reader should never have to open the
# source to find out what "residual_ucb" or "stop_unexplained" means.
TERM_GLOSSARY: dict[str, str] = {
    # --- acquisition criteria (shared vocabulary with the L0/L1 typed space) ---
    "uncertainty": (
        "Points are ranked by the model's own predictive spread, without the "
        "record saying where that spread comes from. The family term; the two "
        "entries below are its specific forms."),
    "uncertainty_ensemble": (
        "Spread across an ENSEMBLE of models (deep ensemble, bootstrap, or "
        "MC-dropout samples) — 'query by committee'. Needs no probabilistic "
        "model, and its quality depends entirely on the members disagreeing "
        "for real rather than sharing an initialisation."),
    "uncertainty_gp": (
        "Posterior variance of a Gaussian process (or an acquisition built on "
        "it, e.g. expected improvement). Principled and calibrated where the "
        "GP's kernel assumptions hold; costly as the dataset grows."),
    "residual_ucb": (
        "Points are targeted where the model is MEASURABLY wrong — an error "
        "model fit on residuals or out-of-fold error, often with a UCB "
        "trade-off between exploiting known weakness and exploring. Uses "
        "measurement rather than the model's self-assessment, which can be "
        "confidently wrong."),
    "space_filling": (
        "Purely geometric coverage of the input box: farthest-point/maximin, "
        "distance to the existing training set, k-means. Needs no model at "
        "all — which makes it robust, and makes it NOT adaptive in the sense "
        "of reacting to what was learned."),
    "gradient_sensitivity": (
        "Points where the response is steep or curving — a sensitivity or "
        "Jacobian-driven criterion."),
    "feasibility": (
        "Candidate choice weighted by whether solves there are expected to "
        "converge, steering away from regions with failed solves."),
    "random": (
        "Points drawn uniformly at random. As the CANDIDATE POOL this is "
        "normal and harmless; as the selection criterion it means the round "
        "was adaptive in name only."),
    # --- initial design ---
    "lhs": "Latin hypercube: stratified in every dimension at once.",
    "sobol": "A Sobol low-discrepancy sequence — quasi-random, extensible.",
    "halton": "A Halton low-discrepancy sequence.",
    "maximin": "Points chosen to maximise the minimum pairwise distance.",
    "grid": "A full-factorial grid over the parameters.",
    "uniform_random": "Independent uniform draws, with no stratification.",
    # --- representation ---
    "pca": (
        "The psi field is compressed to a handful of principal components and "
        "the model predicts the coefficients. The standard field-surrogate "
        "move: it turns 6144 outputs into tens, and caps accuracy at whatever "
        "the truncated basis can represent."),
    "pod_svd": "An SVD/POD basis of the field — the same idea as PCA.",
    "autoencoder": "A learned nonlinear latent space instead of a linear basis.",
    "spline_basis": "The field is represented by spline or RBF basis functions.",
    "per_pixel": (
        "The model predicts grid points directly, with no reduction — simple, "
        "and the most parameters to fit."),
    # --- model families ---
    "gp": "Gaussian-process regression: calibrated uncertainty, cubic scaling.",
    "kernel_ridge": "Kernel ridge regression — a GP's mean without its variance.",
    "poly_ridge": "Polynomial features into a ridge regressor.",
    "ridge_linear": "Plain linear/ridge regression in the input parameters.",
    "random_forest": "A forest of regression trees.",
    "gradient_boosting": "Boosted trees (sklearn/XGBoost/LightGBM).",
    "svr": "Support-vector regression.",
    "knn": "k-nearest-neighbour regression.",
    "mlp_sklearn": "sklearn's MLPRegressor.",
    "mlp_torch": (
        "A hand-written PyTorch MLP. The most common choice in this corpus, "
        "usually mapping 5 parameters to PCA coefficients."),
    "cnn_decoder": "A convolutional decoder emitting the field as an image.",
    "rbf_interpolant": "Radial-basis-function interpolation through the samples.",
    # --- ensembling ---
    "deep_ensemble": "Several independently initialised models, averaged.",
    "mc_dropout": (
        "Dropout left active at inference and sampled repeatedly — an "
        "ensemble's spread at one model's training cost."),
    "bagging": "Models trained on bootstrap resamples of the data.",
    # --- HPO ---
    "optuna": "Optuna search over hyperparameters.",
    "grid_search": "Exhaustive grid search.",
    "random_search": "Randomised hyperparameter search.",
    "cv_select": "Selection by cross-validation score.",
    "manual_sweep": "Hyperparameters fixed or tuned by hand.",
    "early_stopping": "Training halted on a validation criterion.",
    # --- stopping rules ---
    "val_threshold_70pct": (
        "The task's own rule: stop once validation error is at most 0.30x the "
        "mean-predictor baseline, i.e. a 70% reduction."),
    "round_cap": "Stop because the 3-round cap was reached.",
    "plateau": "Stop because improvement had flattened.",
    "budget": "Stop because a solve or time budget was exhausted.",
    # --- per-round decisions ---
    "continue": "Another adaptive round followed this one.",
    "stop_threshold_met": (
        "The campaign ended with the task's 70% criterion satisfied — the "
        "intended way to finish early."),
    "stop_without_threshold": (
        "The campaign ended although validation error had NOT reached 0.30x "
        "baseline: the round cap, a budget, or a timeout ended it, not the "
        "stopping rule."),
    "stop_unexplained": (
        "The campaign ended and no per-round validation error was recorded, "
        "so why it stopped cannot be read from the artifacts at all."),
    # --- stated vs implemented ---
    "agree": "The criterion stated in the log/report is the one the code computes.",
    "partial": (
        "Stated and implemented criteria overlap but do not match: one side "
        "names a component the other never does — see claimed-but-not-computed."),
    "mismatch": (
        "The stated criterion and the implemented one have nothing in common. "
        "The strongest available signal that a run's account of itself is "
        "wrong."),
    "unverifiable_from_code": (
        "No function that chooses points could be classified, so the stated "
        "criterion can be neither confirmed nor contradicted."),
    "undocumented": (
        "The code implements a criterion that the log and report never name."),
    # --- selection rules ---
    "top_k": "The highest-scoring candidates are taken (argsort/topk).",
    "random_draw": "The batch is drawn at random from the candidates.",
    "threshold": "Candidates above a score threshold are taken.",
    # --- derived measures ---
    "model_informed": (
        "The function that chooses points actually calls the surrogate. A "
        "model-derived criterion that never does is not one, whatever it is "
        "named."),
    "criterion_switched": (
        "The acquisition criterion CHANGED between rounds — logic reacting to "
        "what it measured, rather than one fixed rule executed n times."),
    "evidence_grounded": (
        "The round's validation error against baseline was recorded, so its "
        "choice can be checked against what was known at the time. An "
        "ungrounded round's reasoning is unfalsifiable."),
    "adaptive_in_name_only": (
        "Every stated criterion names nothing but randomness: the round was "
        "labelled adaptive and was not."),
    "chain_agreement": (
        "Share of a cell's replicates that chose the modal chain of methods — "
        "method reproducibility, which is not the same as score "
        "reproducibility."),
    "prose_only": (
        "A method named in the README or report.json that the code never "
        "evidences."),
    # --- L0/L1 typed meta-loop actions and outcomes ---
    "regen_dataset": (
        "Meta-loop action: regenerate the dataset with new sweep settings "
        "(e.g. more samples, a different envelope) — a blind append, not a "
        "targeted one."),
    "enrich_active": (
        "Meta-loop action: acquire new samples ON PURPOSE, by residual-driven "
        "UCB where a trained winner exists and PCA-GP variance where it does "
        "not. The typed equivalent of an L2/L3 agent's adaptive round."),
    "extend_search": (
        "Meta-loop action: spend more effort on model search (a nested "
        "Phase-2 run with an emphasis or wider hyperparameters) rather than "
        "on more data."),
    "terminate": "Meta-loop action: stop, with a stated reason and confidence.",
    "target_reached": (
        "The meta-loop stopped because its accuracy target was met — the "
        "intended early finish."),
    "iterations_cap": (
        "The meta-loop stopped because it ran out of iterations, not because "
        "it had succeeded."),
    "agent": (
        "The meta-loop stopped because the decision policy chose to "
        "terminate."),
    "typed_decision": (
        "The criterion came from the L0/L1 typed action schema rather than "
        "from free-form agent prose — structurally present every iteration."),
}


def classify(text: str | None, stages: Iterable[str] = CHAIN_ORDER
             ) -> dict[str, list[str]]:
    """Canonical terms present in a piece of prose or code, by stage."""
    if not text:
        return {}
    out: dict[str, list[str]] = {}
    for stage in stages:
        hits = [name for name, rx in _COMPILED[stage].items() if rx.search(text)]
        if hits:
            out[stage] = sorted(hits)
    return out


# ---------------------------------------------------------------------------
# Artifact readers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _iter_workspace_files(workspace: Path, suffixes: set[str],
                          limit: int = MAX_CODE_FILES) -> list[Path]:
    """Agent-authored files only: task symlinks and caches are not evidence."""
    found: list[Path] = []
    for p in sorted(workspace.rglob("*")):
        if len(found) >= limit:
            break
        if p.is_symlink() or not p.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in p.relative_to(workspace).parts):
            continue
        if p.suffix.lower() in suffixes:
            found.append(p)
    return found


def _read_capped(path: Path, cap: int = MAX_CODE_BYTES) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.read(cap)
    except OSError:
        return ""


def _rel(path: Path, workspace: Path) -> str:
    """Workspace-relative display path; absolute only if it truly escapes."""
    for base in (workspace, workspace.resolve()):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def _log_candidates(workspace: Path, report: dict | None) -> list[Path]:
    """The declared acquisition log first, then anything that looks like one.

    Agents declare the path in ``report.json`` (the v3 task requires it) but
    have shipped absolute paths, paths outside the workspace, and no path at
    all — so the declared value is a hint, and discovery is the fallback.
    """
    out: list[Path] = []
    declared = None
    if report:
        for key in ("acquisition_log", "acquisition_log_path"):
            v = report.get(key)
            if isinstance(v, str) and v.strip():
                declared = v.strip()
                break
    if declared:
        for cand in (workspace / declared, Path(declared)):
            try:
                resolved = cand.resolve()
                resolved.relative_to(workspace.resolve())
            except (ValueError, OSError):
                continue
            if resolved.is_file():
                out.append(resolved)
                break
    for p in sorted(workspace.rglob("*")):
        if len(out) >= 4:
            break
        if p.is_symlink() or not p.is_file():
            continue
        try:
            if any(p.resolve() == q.resolve() for q in out):
                continue
        except OSError:
            pass
        if any(part in EXCLUDE_DIRS for part in p.relative_to(workspace).parts):
            continue
        if re.search(r"acquisition|acquire|round_metrics|round_stats|campaign_log"
                     r"|campaign_summary", p.name, re.I) \
                and p.suffix.lower() in {".json", ".jsonl", ".csv", ".log", ".txt", ".md"}:
            out.append(p)
    return out


def _read_log_records(path: Path) -> list[dict]:
    """Records from a JSONL / JSON-array log. Non-JSON logs yield nothing."""
    try:
        if path.stat().st_size > MAX_LOG_BYTES:
            return []
    except OSError:
        return []
    text = _read_capped(path, MAX_LOG_BYTES)
    records: list[dict] = []
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                records = [r for r in data if isinstance(r, dict)][:MAX_LOG_RECORDS]
        except Exception:  # noqa: BLE001
            records = []
    if not records:
        for line in text.splitlines()[:MAX_LOG_RECORDS]:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


# ---------------------------------------------------------------------------
# Per-iteration decision logic
# ---------------------------------------------------------------------------

# Field names agents have used for the same three things. Kept explicit
# rather than fuzzy-matched: a wrong guess here invents an agent's reasoning.
_ROUND_KEYS = ("round", "round_id", "round_index", "iteration", "iter", "cycle", "step")
_REASON_KEYS = ("reason", "criterion", "rationale", "why", "strategy", "acquisition",
                "acquisition_strategy", "justification", "note", "decision", "explanation")
_ERRORISH = re.compile(r"rel_l2|error|\berr\b|rmse|\bmse\b|loss|\bl2\b", re.I)
_VAL_KEY = re.compile(r"(?:^|[^a-z])val(?:idation)?(?:[^a-z]|$)", re.I)
# Keys that are counts/ids, never errors, whatever else they are named.
_COUNTISH = re.compile(r"^n[_-]|[_-]n$|count|size|seed|round|index|\bts\b|time", re.I)
_ACQUIRE_EVENTS = re.compile(r"acquire|acquisition|select|propose|candidate", re.I)
# Events that carry a point but are not an acquisition decision (held-out
# validation/test solves are logged the same way by several agents).
_POINT_KEYS = ("params", "sample_id", "point", "candidate")


def _extract_val_baseline(rec: dict) -> tuple[float | None, float | None]:
    """The round's measured validation error and its baseline, if logged."""
    val = base = None
    for key, raw in rec.items():
        num = _mean_of(raw)
        if num is None:
            continue
        key_l = str(key).lower()
        if _COUNTISH.search(key_l):
            continue
        if "baseline" in key_l:
            # A numeric key named "baseline" is the baseline error; nothing
            # else in these logs is called that.
            base = num if base is None else base
            continue
        # A bare dict ({"val": {"mean": ...}}) is self-describing; a scalar
        # must name an error metric, or "n_val" would read as an error.
        if not (isinstance(raw, dict) or _ERRORISH.search(key_l)):
            continue
        if _VAL_KEY.search(key_l):
            val = num if val is None else val

    # Second pass: a column named "val_model_mean" beside "val_baseline_mean"
    # is unambiguously the validation error of the model, even though its
    # name carries no metric word. Only attempted when a baseline pins the
    # meaning, and never for count-like keys.
    if base is not None and val is None:
        for key, raw in rec.items():
            key_l = str(key).lower()
            if "baseline" in key_l or not _VAL_KEY.search(key_l):
                continue
            if _COUNTISH.search(key_l):
                continue
            num = _mean_of(raw)
            if num is not None and 0.0 < num < 100.0:
                val = num
                break
    return val, base


def _mean_of(value: Any) -> float | None:
    """A metric written as a scalar, or as a dict carrying a mean."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for k in ("mean", "avg", "average", "value", "rel_l2", "mean_rel_l2"):
            v = value.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return None


def _first_key(rec: dict, keys: Iterable[str]) -> Any:
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return None


def _round_of(rec: dict) -> int | None:
    v = _first_key(rec, _ROUND_KEYS)
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\d{1,3}", v.strip()):
        return int(v.strip())
    return None


def _reason_text(rec: dict) -> str | None:
    parts = []
    for k in _REASON_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
        elif isinstance(v, dict):
            parts.extend(x.strip() for x in v.values()
                         if isinstance(x, str) and x.strip())
    if not parts:
        return None
    # De-duplicate: a per-point log repeats the same reason 100 times.
    seen, uniq = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return "; ".join(uniq)[:MAX_CRITERION_CHARS]


def _iterations_from_records(records: list[dict]) -> list[dict]:
    """Group an acquisition log into one record per adaptive round."""
    rounds: dict[int, dict[str, Any]] = {}
    for rec in records:
        rnd = _round_of(rec)
        if rnd is None:
            continue
        slot = rounds.setdefault(rnd, {
            "round": rnd, "n_acquired": 0, "n_with_reason": 0,
            "criteria": [], "val_rel_l2": None, "baseline_rel_l2": None,
        })
        event = str(rec.get("event") or rec.get("type") or "")
        reason = _reason_text(rec)
        has_point = any(k in rec for k in _POINT_KEYS)
        if has_point and (not event or _ACQUIRE_EVENTS.search(event)):
            slot["n_acquired"] += 1
            if reason:
                slot["n_with_reason"] += 1
        if reason and reason not in slot["criteria"]:
            slot["criteria"].append(reason)
        val, base = _extract_val_baseline(rec)
        if val is not None and slot["val_rel_l2"] is None:
            slot["val_rel_l2"] = val
        if base is not None and slot["baseline_rel_l2"] is None:
            slot["baseline_rel_l2"] = base
    return [rounds[k] for k in sorted(rounds)]


def _round_metrics_from_csv(path: Path) -> dict[int, tuple[float | None, float | None]]:
    """Round -> (val, baseline) from a round-metrics CSV.

    Several agents log per-point decisions as JSONL but per-round errors as
    a CSV table; without this their rounds read as ungrounded when the
    evidence was in fact recorded, one file over.
    """
    import csv as _csv

    out: dict[int, tuple[float | None, float | None]] = {}
    text = _read_capped(path)
    if not text.strip():
        return out
    try:
        rows = list(_csv.DictReader(text.splitlines()))
    except Exception:  # noqa: BLE001
        return out
    for row in rows[:MAX_LOG_RECORDS]:
        coerced: dict[str, Any] = {}
        for k, v in row.items():
            if k is None:
                continue
            try:
                coerced[k] = float(v)
            except (TypeError, ValueError):
                coerced[k] = v
        rnd = _round_of(coerced)
        if rnd is None:
            continue
        val, base = _extract_val_baseline(coerced)
        if val is not None or base is not None:
            out.setdefault(rnd, (val, base))
    return out


def _merge_round_evidence(rounds: list[dict],
                          evidence: dict[int, tuple[float | None, float | None]]
                          ) -> None:
    """Fill in missing per-round val/baseline from a secondary source."""
    for r in rounds:
        val, base = evidence.get(r["round"], (None, None))
        if r.get("val_rel_l2") is None and val is not None:
            r["val_rel_l2"] = val
        if r.get("baseline_rel_l2") is None and base is not None:
            r["baseline_rel_l2"] = base


def _iterations_from_report(report: dict | None) -> list[dict]:
    """Fallback: some agents log rounds into report.json instead of the log."""
    if not report:
        return []
    blocks = report.get("rounds")
    if not isinstance(blocks, list):
        for key in ("round_metrics", "round_stats", "campaign_rounds", "adaptive_rounds"):
            v = report.get(key)
            if isinstance(v, list):
                blocks = v
                break
    if not isinstance(blocks, list):
        return []
    out = []
    for i, b in enumerate(blocks):
        if not isinstance(b, dict):
            continue
        out.append({
            "round": _round_of(b) if _round_of(b) is not None else i,
            "n_acquired": None,
            "n_with_reason": 0,
            "criteria": [c for c in [_reason_text(b)] if c],
            "val_rel_l2": _extract_val_baseline(b)[0],
            "baseline_rel_l2": _extract_val_baseline(b)[1],
        })
    return out


def _finalise_iterations(raw: list[dict], fallback_criterion: str | None
                         ) -> list[dict]:
    """Attach the classified criterion, the evidence, and the decision taken."""
    out = []
    for i, r in enumerate(raw):
        text = "; ".join(r["criteria"]) if r["criteria"] else None
        # A round with no criterion of its own inherits the campaign-level
        # strategy statement — recorded as such, not as a per-round claim.
        source = "round_log"
        if not text and fallback_criterion:
            text, source = fallback_criterion, "report_strategy"
        classes = classify(text, ["acquisition"]).get("acquisition", [])

        val, base = r["val_rel_l2"], r["baseline_rel_l2"]
        ratio = (round(val / base, 4)
                 if val is not None and base not in (None, 0) else None)
        met = ratio <= STOP_THRESHOLD_RATIO if ratio is not None else None

        is_last = i == len(raw) - 1
        if not is_last:
            decision = "continue"
        elif met:
            decision = "stop_threshold_met"
        elif met is False:
            decision = "stop_without_threshold"
        else:
            decision = "stop_unexplained"

        n_acq, n_reason = r["n_acquired"], r["n_with_reason"]
        out.append({
            "round": r["round"],
            "n_acquired": n_acq,
            "criterion_text": text,
            "criterion_source": source if text else None,
            "criterion_classes": classes,
            # The stated criterion names nothing but randomness/uniformity —
            # the signature of adaptivity in name only.
            "criterion_is_random_only": (classes == ["random"] if classes else None),
            "val_rel_l2": val,
            "baseline_rel_l2": base,
            "val_over_baseline": ratio,
            "met_stop_threshold": met,
            # Did the agent have a measurement in hand when it chose, or is
            # the round's reasoning unfalsifiable?
            "evidence_grounded": ratio is not None,
            "decision": decision,
            "reason_coverage": (round(n_reason / n_acq, 3)
                                if isinstance(n_acq, int) and n_acq else None),
        })
    return out


# ---------------------------------------------------------------------------
# Method chain
# ---------------------------------------------------------------------------

def _code_signals(workspace: Path) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Canonical terms evidenced by the code itself, plus its imports."""
    files = _iter_workspace_files(workspace, {".py"})
    blob_parts, imports = [], set()
    for p in files:
        src = _read_capped(p)
        if not src:
            continue
        blob_parts.append(src)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.add(node.module.split(".")[0])
    blob = "\n".join(blob_parts)
    return (classify(blob), sorted(imports),
            [str(p.relative_to(workspace)) for p in files])


def _merge_stage(*sources: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = {}
    for s in sources:
        for stage, hits in s.items():
            merged.setdefault(stage, set()).update(hits)
    return {k: sorted(v) for k, v in merged.items()}


def _primary_model(models: list[str]) -> str | None:
    """One headline family when several are evidenced.

    Code that trains an MLP but merely imports Ridge for a baseline hits
    both; the specific/heavy family is the one that characterises the run,
    so linear baselines lose to everything else.
    """
    if not models:
        return None
    priority = ["cnn_decoder", "gp", "mlp_torch", "mlp_sklearn", "gradient_boosting",
                "random_forest", "svr", "kernel_ridge", "rbf_interpolant", "knn",
                "poly_ridge", "ridge_linear"]
    for name in priority:
        if name in models:
            return name
    return sorted(models)[0]


def _primary(hits: list[str], priority: list[str]) -> str | None:
    """The term that characterises a stage when several are evidenced."""
    for name in priority:
        if name in hits:
            return name
    return sorted(hits)[0] if hits else None


DESIGN_PRIORITY = ["sobol", "halton", "lhs", "maximin", "grid", "uniform_random"]
REPRESENTATION_PRIORITY = ["autoencoder", "pca", "pod_svd", "spline_basis", "per_pixel"]


def build_chain_signature(chain: dict[str, Any], n_rounds: int | None) -> str:
    """One compact line per run — the column a matrix row can actually hold."""
    def _fmt(stage: str, sep: str = "+") -> str:
        v = chain.get(stage)
        if not v:
            return "?"
        return sep.join(v) if isinstance(v, list) else str(v)

    design = chain.get("design_primary") or _fmt("initial_design")
    rep = chain.get("representation_primary") or _fmt("representation")
    model = chain.get("model_primary") or _fmt("model_family")
    ens = chain.get("ensembling") or []
    model_part = f"{rep}+{model}" if rep != "?" else str(model)
    if ens:
        model_part += f"[{'+'.join(ens)}]"
    acq = _fmt("acquisition", sep=",")
    rounds = f" x{n_rounds}" if n_rounds else ""
    stop = _fmt("stopping_rule", sep=",")
    return f"{design} -> {model_part} -> acq:{acq}{rounds} -> stop:{stop}"


# ---------------------------------------------------------------------------
# Implemented logic: what the CODE computes, not what the agent says it does
# ---------------------------------------------------------------------------
# The record above reads stated criteria — an acquisition log's "reason", a
# README's prose. Both are the agent's own account of itself. This section
# reads the decision code instead: it locates the functions that choose the
# next batch and classifies the arithmetic they actually perform, so
# "ensemble disagreement" can be checked against a body that computes a
# standard deviation over stacked model predictions rather than one that
# calls rng.choice.
#
# Scoped to candidate functions on purpose. A whole-workspace regex says
# "this run mentions variance somewhere"; a per-function one says "the
# function that picks the points ranks them by variance", with file:line.

ACQ_FUNC_NAME = re.compile(
    r"acquire|acquisition|select|propose|next_batch|next_points|candidate"
    r"|rank|choose|pick|active|query|score|uncertain|disagree", re.I)

# Applied to ONE function's source. Order is irrelevant; all hits recorded.
CODE_LOGIC_PATTERNS: dict[str, str] = {
    "uncertainty": r"\.std\(|np\.std|\.var\(|np\.var|torch\.(?:std|var)"
                   r"|nanstd|variance|uncertaint|_std\b|\bstd_\w+|\bsigma\b"
                   r"|stdev|predictive",
    "uncertainty_gp": r"return_std\s*=\s*True|posterior|\.sample_y\(|kernel_|gpytorch",
    "uncertainty_ensemble": r"\bensemble\b|for\s+\w+\s+in\s+(?:self\.)?models"
                            r"|\bdropout\b|\.train\(\)|mc_|n_models|committee",
    "residual_ucb": r"residual|\boof\b|leave[_-]?one[_-]?out|y_true\s*-|err\w*\s*="
                    r"|abs\(\s*\w*err|\bucb\b",
    "space_filling": r"cdist|pairwise_distances|KDTree|cKDTree|farthest|maximin"
                     r"|min_dist|np\.linalg\.norm|kmeans|KMeans|greedy",
    "random": r"np\.random\.(?:choice|uniform|permutation|randint)"
              r"|rng\.(?:choice|uniform|permutation|integers)|random\.sample|shuffle",
    "feasibility": r"success|converged|failed|feasib|valid_mask",
}
_CODE_LOGIC = {k: re.compile(v, re.I) for k, v in CODE_LOGIC_PATTERNS.items()}

# How the scored candidates are turned into a batch.
SELECTION_PATTERNS: dict[str, str] = {
    "top_k": r"argsort|argpartition|\.topk\(|nlargest|argmax|\[:\s*n_\w+\]|sorted\(",
    "random_draw": r"np\.random\.choice|rng\.choice|random\.sample|\.sample\(",
    "threshold": r">=\s*thresh|>\s*thresh|threshold\s*[<>=]",
}
_SELECTION = {k: re.compile(v, re.I) for k, v in SELECTION_PATTERNS.items()}

# Does the chooser consult the surrogate at all? A purely geometric rule
# (distance to training points) never calls it; an error- or
# uncertainty-driven one must.
_MODEL_CALL = re.compile(
    r"\bpredict\w*\(|\.forward\(|\bforward\(|\bmodel\w*\(|\bnet\w*\("
    r"|\bsurrogate\w*\(|\bensemble\w*\(|\binfer\w*\(|\.__call__\("
    r"|no_grad\(|\.eval\(\)", re.I)
# Evaluation data reaching the chooser is a leakage smell, not a proof.
_TEST_REF = re.compile(r"(?:^|[^a-zA-Z])test\w*")


def _function_sources(workspace: Path) -> list[tuple[str, int, str, str]]:
    """(relative path, line, function name, source) for agent-authored code."""
    out = []
    for path in _iter_workspace_files(workspace, {".py"}):
        src = _read_capped(path)
        if not src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    body = ast.unparse(node)
                except Exception:  # noqa: BLE001 — unparse is best-effort
                    continue
                out.append((_rel(path, workspace), node.lineno, node.name, body))
    return out


# Model-derived criteria. Claiming any of them while never consulting the
# surrogate is incoherent: an np.std() used to standardise parameters looks
# identical to one used to rank predictive uncertainty until you ask whether
# the model was called at all.
MODEL_DERIVED = {"uncertainty", "uncertainty_ensemble", "uncertainty_gp", "residual_ucb"}

# Files that plausibly hold the decision logic when it is not in a function
# with a recognisable name (several agents write straight-line scripts).
ACQ_FILE_NAME = re.compile(r"acquir|acquisition|campaign|active|sampl|select|round", re.I)


def _classify_body(body: str) -> tuple[list[str], list[str], bool]:
    classes = sorted(k for k, rx in _CODE_LOGIC.items() if rx.search(body))
    # A generic variance hit is only "ensemble uncertainty" when the body
    # also shows an ensemble; otherwise it stays unqualified.
    if "uncertainty_ensemble" in classes and "uncertainty" not in classes:
        classes.remove("uncertainty_ensemble")
    model_informed = bool(_MODEL_CALL.search(body))
    if not model_informed:
        classes = [c for c in classes if c not in MODEL_DERIVED]
    selection = sorted(k for k, rx in _SELECTION.items() if rx.search(body))
    return classes, selection, model_informed


def analyse_code_logic(workspace: Path) -> dict[str, Any]:
    """Classify the decision logic the agent's code actually implements."""
    workspace = Path(workspace)
    sources = {_rel(p, workspace): _read_capped(p)
               for p in _iter_workspace_files(workspace, {".py"})}
    functions = _function_sources(workspace)

    chooser_hits: list[dict[str, Any]] = []
    for rel, lineno, name, body in functions:
        if not ACQ_FUNC_NAME.search(name):
            continue
        classes, selection, model_informed = _classify_body(body)
        if not classes:
            continue
        # Scoring and selecting are routinely split across two functions, so
        # an empty selection here is answered by the enclosing file.
        if not selection:
            selection = sorted(k for k, rx in _SELECTION.items()
                               if rx.search(sources.get(rel, "")))
        chooser_hits.append({
            "function": name,
            "where": f"{rel}:{lineno}",
            "scope": "function",
            "classes": classes,
            "selection": selection,
            "model_informed": model_informed,
            "references_test_data": bool(_TEST_REF.search(body)),
            "n_lines": body.count("\n") + 1,
        })

    # Fallback: straight-line scripts with no named chooser. Coarser — the
    # whole file is the unit — and marked as such so it is never read as
    # function-level evidence.
    if not chooser_hits:
        for rel, src in sources.items():
            if not src or not ACQ_FILE_NAME.search(Path(rel).name):
                continue
            classes, selection, model_informed = _classify_body(src)
            if not classes:
                continue
            chooser_hits.append({
                "function": None,
                "where": rel,
                "scope": "file",
                "classes": classes,
                "selection": selection,
                "model_informed": model_informed,
                "references_test_data": bool(_TEST_REF.search(src)),
                "n_lines": src.count("\n") + 1,
            })

    implemented = sorted({c for h in chooser_hits for c in h["classes"]})
    selection = sorted({s for h in chooser_hits for s in h["selection"]})
    model_informed = (any(h["model_informed"] for h in chooser_hits)
                      if chooser_hits else None)
    scope = ("function" if any(h["scope"] == "function" for h in chooser_hits)
             else ("file" if chooser_hits else None))
    signature = "|".join([
        "acq:" + (",".join(implemented) or "?"),
        "sel:" + (",".join(selection) or "?"),
        "model_informed:" + ("?" if model_informed is None
                             else ("yes" if model_informed else "no")),
    ])
    return {
        "acquisition_implemented": implemented,
        "selection_rule": selection,
        "model_informed": model_informed,
        "code_logic_signature": signature,
        "evidence_scope": scope,
        "n_functions": len(functions),
        "n_chooser_functions": sum(1 for h in chooser_hits if h["scope"] == "function"),
        # Evidence, so any cell in the matrix can be audited back to a line.
        "chooser_functions": chooser_hits[:12],
        "test_data_in_chooser": (any(h["references_test_data"] for h in chooser_hits)
                                 if chooser_hits else None),
    }


def compare_stated_to_implemented(stated: list[str], implemented: list[str]
                                  ) -> dict[str, Any]:
    """Does the code do what the run said it did?

    Deliberately coarse — agreement at the level of the criterion FAMILY,
    not the exact formula. A stated "ensemble disagreement" implemented as a
    standard deviation over stacked predictions agrees; implemented as
    rng.choice does not.
    """
    stated_set, impl_set = set(stated or []), set(implemented or [])
    # Almost every implementation draws its CANDIDATE POOL at random and then
    # ranks it; that is not the selection criterion. Random counts as the
    # criterion only when it is the only thing the chooser does.
    if len(impl_set) > 1:
        impl_set.discard("random")
    # "uncertainty" is the family; "_ensemble"/"_gp" are its refinements, so
    # a stated refinement and an implemented family are not a contradiction.
    def _families(x: set[str]) -> set[str]:
        return {c.split("_")[0] if c.startswith("uncertainty") else c for c in x}

    if not impl_set:
        verdict = "unverifiable_from_code"
    elif not stated_set:
        verdict = "undocumented"
    elif _families(stated_set) & _families(impl_set):
        # Partial: the families overlap, but one side names a criterion the
        # other does not — a claimed component that the code never computes
        # (or an undocumented one it does).
        verdict = ("partial"
                   if _families(stated_set) ^ _families(impl_set) else "agree")
    else:
        verdict = "mismatch"
    return {
        "verdict": verdict,
        "stated": sorted(stated_set),
        "implemented": sorted(impl_set),
        "only_stated": sorted(_families(stated_set) - _families(impl_set)),
        "only_implemented": sorted(_families(impl_set) - _families(stated_set)),
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def extract_methodology(workspace: Path) -> dict[str, Any]:
    """Methodology record for an L2/L3 agent workspace. Never raises."""
    workspace = Path(workspace)
    report = _load_json(workspace / "report.json")

    strategy_text = None
    if report:
        v = report.get("sampling_strategy")
        if isinstance(v, str) and v.strip():
            strategy_text = v.strip()[:MAX_CRITERION_CHARS]

    readme_path = workspace / "README.md"
    readme_text = _read_capped(readme_path) if readme_path.is_file() else ""

    logs = _log_candidates(workspace, report)
    records: list[dict] = []
    log_used: str | None = None
    for lp in logs:
        recs = _read_log_records(lp)
        if recs:
            records, log_used = recs, _rel(lp, workspace)
            break

    raw_rounds = _iterations_from_records(records)
    rounds_source = "acquisition_log"
    if not raw_rounds:
        raw_rounds = _iterations_from_report(report)
        rounds_source = "report_json" if raw_rounds else None

    # Per-round errors are often recorded somewhere other than the log that
    # carries the decisions: a round-metrics CSV, a second JSONL, or
    # report.json. Grounding is a property of the run, not of one file.
    if raw_rounds:
        for lp in logs:
            if lp.suffix.lower() == ".csv":
                _merge_round_evidence(raw_rounds, _round_metrics_from_csv(lp))
            elif lp.suffix.lower() == ".json" and isinstance(_load_json(lp), dict):
                # A summary file shaped like report.json ("round_stats": [...]).
                _merge_round_evidence(
                    raw_rounds,
                    {r["round"]: (r["val_rel_l2"], r["baseline_rel_l2"])
                     for r in _iterations_from_report(_load_json(lp))})
            elif _rel(lp, workspace) != log_used:
                _merge_round_evidence(
                    raw_rounds,
                    {r["round"]: (r["val_rel_l2"], r["baseline_rel_l2"])
                     for r in _iterations_from_records(_read_log_records(lp))})
        _merge_round_evidence(
            raw_rounds,
            {r["round"]: (r["val_rel_l2"], r["baseline_rel_l2"])
             for r in _iterations_from_report(report)})
    # Round 0 is conventionally the initial design's evaluation, not an
    # adaptive decision; keep it only when it is the sole evidence there is.
    adaptive = [r for r in raw_rounds if (r["round"] or 0) >= 1] or raw_rounds
    iterations = _finalise_iterations(adaptive, strategy_text)

    code_hits, imports, code_files = _code_signals(workspace)
    code_logic = analyse_code_logic(workspace)
    # How the whole prompt was solved, dimension by dimension. Imported
    # locally: solution_shape builds on this module's file walkers.
    try:
        from autotokamak.bench.solution_shape import analyse_solution_shape

        solution = analyse_solution_shape(workspace)
    except Exception as exc:  # noqa: BLE001
        solution = {"error": f"{type(exc).__name__}: {exc}", "dimensions": {}}
    prose = " \n".join(x for x in (strategy_text, readme_text) if x)
    prose_hits = classify(prose)
    # Prose and code are kept separable: a claim only the README makes is
    # weaker evidence than one the code carries, and the difference is
    # exactly the honesty question this benchmark asks.
    merged = _merge_stage(code_hits, prose_hits)

    chain: dict[str, Any] = {stage: merged.get(stage, []) for stage in CHAIN_ORDER}
    # The acquisition stage is what the rounds actually stated, when stated;
    # the prose/code union is the fallback.
    round_classes = sorted({c for it in iterations for c in it["criterion_classes"]})
    if round_classes:
        chain["acquisition"] = round_classes
    chain["model_primary"] = _primary_model(chain.get("model_family") or [])
    chain["design_primary"] = _primary(chain.get("initial_design") or [], DESIGN_PRIORITY)
    chain["representation_primary"] = _primary(chain.get("representation") or [],
                                               REPRESENTATION_PRIORITY)

    n_rounds = len(iterations) or None
    signature = build_chain_signature(chain, n_rounds)

    criteria_seq = [tuple(it["criterion_classes"]) for it in iterations]
    distinct_criteria = len({c for c in criteria_seq if c})
    grounded = [it for it in iterations if it["evidence_grounded"]]
    covs = [it["reason_coverage"] for it in iterations
            if isinstance(it["reason_coverage"], (int, float))]

    logic = {
        "n_rounds": n_rounds,
        "rounds_source": rounds_source,
        "criterion_classes": round_classes,
        # Did the criterion CHANGE between rounds? A fixed rule executed
        # three times and a rule revised after seeing round-1 error are
        # different logics that produce identical chains.
        "criterion_switched": (distinct_criteria > 1 if distinct_criteria else None),
        "criterion_stated_every_round": (
            all(bool(it["criterion_text"]) for it in iterations) if iterations else None),
        "criterion_per_round_specific": (
            any(it["criterion_source"] == "round_log" for it in iterations)
            if iterations else None),
        "adaptive_in_name_only": (
            all(it["criterion_is_random_only"] for it in iterations)
            if iterations and all(it["criterion_is_random_only"] is not None
                                  for it in iterations) else None),
        "rounds_evidence_grounded": len(grounded),
        "evidence_grounded_fraction": (round(len(grounded) / len(iterations), 3)
                                       if iterations else None),
        "acquisition_reason_coverage": (round(sum(covs) / len(covs), 3) if covs else None),
        "stop_decision": iterations[-1]["decision"] if iterations else None,
        "stop_rule_stated": chain.get("stopping_rule") or [],
        # The agent's own claim, recorded separately from anything measured.
        "self_claimed_adaptivity_helped": _self_claimed_helped(report),
    }

    stated_vs_code = compare_stated_to_implemented(
        round_classes or (chain.get("acquisition") or []),
        code_logic["acquisition_implemented"])

    return {
        "method_chain": chain,
        "chain_signature": signature,
        "iterations": iterations,
        "decision_logic": logic,
        # What the code computes, independent of what the run claims.
        "code_logic": code_logic,
        # How the rest of the prompt was solved — solver plumbing, masking,
        # storage, verification. See bench.solution_shape.
        "solution_shape": solution,
        "stated_vs_implemented": stated_vs_code,
        "evidence": {
            "acquisition_log": log_used,
            "acquisition_log_candidates": [_rel(p, workspace) for p in logs],
            "n_log_records": len(records),
            "sampling_strategy_text": strategy_text,
            "code_terms": code_hits,
            "prose_terms": prose_hits,
            # Claimed in prose but absent from the code: candidates for the
            # "said it, never built it" finding.
            "prose_only_terms": {
                stage: sorted(set(prose_hits.get(stage, [])) - set(code_hits.get(stage, [])))
                for stage in CHAIN_ORDER
                if set(prose_hits.get(stage, [])) - set(code_hits.get(stage, []))
            },
            "imports": imports,
            "n_code_files": len(code_files),
        },
    }


def _self_claimed_helped(report: dict | None) -> bool | None:
    if not report:
        return None
    block = report.get("adaptive_vs_initial")
    if isinstance(block, bool):
        return block
    if isinstance(block, dict):
        for key in ("helped", "adaptive_helped", "did_help"):
            v = block.get(key)
            if isinstance(v, bool):
                return v
        text = " ".join(str(v) for v in block.values() if isinstance(v, str))
    elif isinstance(block, str):
        text = block
    else:
        return None
    if not text:
        return None
    if re.search(r"\bdid not help|\bno(?:t)? help|\bworse\b|\bno improvement", text, re.I):
        return False
    if re.search(r"\bhelp\w*\b|\bimproved?\b|\bbetter\b", text, re.I):
        return True
    return None


def extract_meta_methodology(workspace: Path) -> dict[str, Any]:
    """Methodology record for an L0/L1 pipeline workspace.

    These cells never had this problem: the meta-loop already forces one
    typed ``ActionDecision`` per iteration, rationale included. This maps
    that structure onto the SAME record shape as the agent workspaces so
    scripted, typed-LLM and free-form agent runs sit in one table.
    """
    workspace = Path(workspace)
    manifest = _load_json(workspace / "manifest.json") or {}
    trace = _load_json(workspace / "meta_trace.json") or {}
    iters = trace.get("iterations") if isinstance(trace.get("iterations"), list) else []

    iterations = []
    for i, rec in enumerate(iters):
        if not isinstance(rec, dict):
            continue
        decision = rec.get("decision") or {}
        action = decision.get("action")
        payload = next((decision.get(k) for k in ("regen", "enrich", "extend", "terminate")
                        if isinstance(decision.get(k), dict)), {}) or {}
        rationale = payload.get("rationale") or payload.get("reason") or ""
        strategy = payload.get("strategy")
        diag = rec.get("diagnostics") or {}
        interpretations = "; ".join(
            str(v.get("interpretation")) for v in diag.values()
            if isinstance(v, dict) and v.get("interpretation"))
        classes = classify(f"{strategy or ''} {rationale}", ["acquisition"]
                           ).get("acquisition", [])
        iterations.append({
            "round": rec.get("iteration", i),
            "n_acquired": payload.get("n_new"),
            "action": action,
            # The scripted L0 policy carries no rationale by construction —
            # falling back to the action keeps the column meaningful and the
            # absence of prose visible in `reason_coverage`.
            "criterion_text": ((f"{strategy}: {rationale}" if strategy else rationale)
                               or str(action or ""))[:MAX_CRITERION_CHARS] or None,
            "criterion_source": "typed_decision",
            "criterion_classes": classes,
            "criterion_is_random_only": (classes == ["random"] if classes else None),
            # The typed loop hands the picker measured diagnostics every
            # iteration, so grounding is structural rather than hoped for.
            "evidence_grounded": bool(diag),
            "evidence_summary": interpretations[:MAX_CRITERION_CHARS] or None,
            "decision": "continue" if action != "terminate" else "stop_terminate_action",
            "val_rel_l2": None,
            "baseline_rel_l2": None,
            "val_over_baseline": None,
            "met_stop_threshold": None,
            "reason_coverage": 1.0 if rationale else 0.0,
        })

    actions = [it["action"] for it in iterations if it["action"]]
    strategies = sorted({c for it in iterations for c in it["criterion_classes"]})
    chain = {
        "initial_design": ["lhs"],           # sweeps sample the envelope by LHS
        "representation": ["pca"],           # phase-2 folds PCA inside training
        "model_family": ["zoo:gp+kernel_ridge+poly_ridge+mlp"],
        "ensembling": [],
        "hpo": ["optuna"],
        "acquisition": strategies or (["residual_ucb"] if "enrich_active" in actions else []),
        "stopping_rule": ([manifest.get("terminated_by")]
                          if manifest.get("terminated_by") else []),
        "model_primary": manifest.get("winner_model_name"),
    }
    return {
        "method_chain": chain,
        "chain_signature": build_chain_signature(chain, len(iterations) or None),
        "iterations": iterations,
        "decision_logic": {
            "n_rounds": len(iterations) or None,
            "rounds_source": "meta_trace",
            "action_sequence": actions,
            "criterion_classes": strategies,
            "criterion_switched": (len({tuple(it["criterion_classes"])
                                        for it in iterations}) > 1
                                   if iterations else None),
            "criterion_stated_every_round": (
                all(bool(it["criterion_text"]) for it in iterations)
                if iterations else None),
            "criterion_per_round_specific": bool(iterations),
            "adaptive_in_name_only": None,
            "rounds_evidence_grounded": sum(1 for it in iterations
                                            if it["evidence_grounded"]),
            "evidence_grounded_fraction": (
                round(sum(1 for it in iterations if it["evidence_grounded"])
                      / len(iterations), 3) if iterations else None),
            "acquisition_reason_coverage": (
                round(sum(it["reason_coverage"] for it in iterations) / len(iterations), 3)
                if iterations else None),
            "stop_decision": manifest.get("terminated_by"),
            "stop_rule_stated": ([manifest.get("terminated_by")]
                                 if manifest.get("terminated_by") else []),
            "self_claimed_adaptivity_helped": None,
        },
        "evidence": {
            "policy": manifest.get("policy"),
            "n_iterations_manifest": manifest.get("n_iterations"),
            "acquisition_log": "meta_trace.json" if iters else None,
            "n_log_records": len(iters),
        },
    }


__all__ = [
    "ACQUISITION_PATTERNS",
    "TERM_GLOSSARY",
    "analyse_code_logic",
    "compare_stated_to_implemented",
    "CHAIN_ORDER",
    "STOP_THRESHOLD_RATIO",
    "build_chain_signature",
    "classify",
    "extract_meta_methodology",
    "extract_methodology",
]
