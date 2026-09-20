# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""How each agent SOLVED the prompt — the whole approach, not just the round loop.

``bench.methodology`` answers "what did it choose, and why, at each adaptive
round". That is one paragraph of a task that asks for an entire pipeline:
drive a finite-element solver under a singleton constraint, run and validate a
data campaign, mask a field that is undefined outside the plasma, map a
triangular mesh onto a fixed rectangular grid, train a surrogate, and ship a
CLI that a stranger can run. Two agents can share an acquisition criterion and
still have solved almost none of the same problems the same way.

This module reads the code for those decisions, one dimension at a time, and
returns them in a form that transposes: dimension as row, condition as column,
which is what a cross-comparison actually is.

Why scoped detection, not a keyword sweep
-----------------------------------------
A whole-workspace grep for "subprocess" is true of every run in this corpus —
including the ones that solve in-process and merely shell out to check a
deliverable. So every dimension here is answered from the code that plays the
relevant ROLE: the OFT strategy from the file that constructs ``OFT_env``, the
masking rule from ``predict.py`` and the solver wrapper, the self-test from
whatever actually invokes ``predict.py`` as a subprocess. Each answer carries
``file:line`` so a disagreement is settled by reading the line, not by
re-running the tool.

Unknown stays unknown. A dimension nothing evidences is ``None``, never a
default that would quietly make two agents look alike.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from autotokamak.bench.methodology import (
    _iter_workspace_files,
    _read_capped,
    _rel,
)

MAX_EVIDENCE = 3

# The task MANDATES a meshing milestone, a pilot and a deliverable self-test,
# so every workspace contains demo/verification scripts that are not the
# pipeline. Attributing "this run solves in subprocesses" to a repro-check
# script is how a cross-comparison quietly becomes fiction, so the production
# path is the default scope and these are excluded from it.
NON_PRODUCTION = re.compile(
    r"(?:^|/)(?:tests?|scratch)/|smoke|milestone|preflight|parity|repro|purity"
    r"|integrity|_inspect|(?:^|/)tests?_|test_[^/]*\.py$|[^/]*_test\.py$", re.I)


class CodeIndex:
    """Agent-authored Python, indexed so a match can name its function."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.files: dict[str, str] = {}
        # Production path = everything the pipeline actually runs on.
        self.production: list[str] = []
        # rel -> [(start_line, end_line, name)], innermost last
        self._funcs: dict[str, list[tuple[int, int, str]]] = {}
        for path in _iter_workspace_files(self.workspace, {".py"}):
            src = _read_capped(path)
            if not src:
                continue
            rel = _rel(path, self.workspace)
            self.files[rel] = src
            spans = []
            try:
                tree = ast.parse(src)
            except SyntaxError:
                self._funcs[rel] = []
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    end = getattr(node, "end_lineno", node.lineno)
                    spans.append((node.lineno, end, node.name))
            self._funcs[rel] = sorted(spans)
        self.production = [rel for rel in self.files
                           if not NON_PRODUCTION.search(rel)] or list(self.files)

    def enclosing(self, rel: str, line: int) -> str | None:
        best = None
        for start, end, name in self._funcs.get(rel, []):
            if start <= line <= end:
                best = name  # later spans are nested deeper
        return best

    def search(self, pattern: str, files: list[str] | None = None,
               *, everywhere: bool = False) -> list[dict[str, Any]]:
        """Matching lines. Defaults to the production path, not every file."""
        rx = re.compile(pattern, re.I)
        hits = []
        scope = (list(self.files) if everywhere
                 else (files if files is not None else self.production))
        for rel in scope:
            src = self.files.get(rel)
            if not src:
                continue
            for i, line in enumerate(src.splitlines(), start=1):
                if rx.search(line):
                    hits.append({"where": f"{rel}:{i}", "file": rel, "line": i,
                                 "text": line.strip()[:160],
                                 "function": self.enclosing(rel, i)})
        return hits

    def files_matching(self, pattern: str, *, everywhere: bool = False) -> list[str]:
        rx = re.compile(pattern, re.I)
        return [rel for rel in (self.files if everywhere else self.production)
                if rx.search(rel)]

    def files_containing(self, pattern: str, *, everywhere: bool = False) -> list[str]:
        rx = re.compile(pattern, re.I)
        return [rel for rel in (self.files if everywhere else self.production)
                if rx.search(self.files.get(rel, ""))]

    def function_source(self, rel: str, name: str) -> str:
        src = self.files.get(rel, "")
        lines = src.splitlines()
        for start, end, fname in self._funcs.get(rel, []):
            if fname == name:
                return "\n".join(lines[start - 1:end])
        return ""


def _dim(value: str | None, hits: list[dict], note: str = "") -> dict[str, Any]:
    return {
        "value": value,
        "evidence": [h["where"] for h in hits[:MAX_EVIDENCE]],
        "note": note or None,
    }


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

def _oft_strategy(idx: CodeIndex) -> dict[str, Any]:
    """The task names one hard constraint: one OFT_env per process, ever.

    Three answers appear in this corpus, and they are not stylistic — they
    decide whether the campaign can parallelise at all, and whether a single
    bad solve takes the whole run down with it.
    """
    hits = idx.search(r"OFT_env\s*\(")
    if not hits:
        return _dim(None, [], "no OFT_env construction found")

    owners = {h["file"] for h in hits}
    # Does a worker process own it (fork/spawn per batch), or the main one?
    child = idx.search(r"multiprocessing|ProcessPoolExecutor|fork|spawn|Pool\s*\(")
    child_files = {h["file"] for h in child}
    # Singleton bookkeeping: a cached instance, a module-level global, or a
    # class that guards construction.
    singleton = [h for h in idx.search(
        r"_instance|_ENV\b|_oft_env|global\s+\w*env|@lru_cache|@cache\b"
        r"|if\s+\w*env\w*\s+is\s+None")
        if h["file"] in owners]

    per_solve = [h for h in hits
                 if (h["function"] or "").lower() in {"solve", "run_solve", "solve_one",
                                                      "_solve", "solve_case"}]
    if child_files & owners:
        value = "per_worker_process"
    elif singleton:
        value = "singleton_reused"
    elif per_solve:
        value = "per_solve"
    else:
        value = "module_level_once"
    return _dim(value, hits + singleton[:1],
                f"constructed in {len(owners)} file(s)")


def _solve_isolation(idx: CodeIndex) -> dict[str, Any]:
    """How a solve is insulated from the one before it.

    Scoped to the code that DRIVES solves. Every workspace shells out
    somewhere — to run its own self-test, if nothing else — and reading that
    as the campaign's execution model is simply wrong.
    """
    scope = list(dict.fromkeys(
        idx.files_containing(r"OFT_env\s*\(|import\s+\w*solver|from\s+\w*solver")
        + idx.files_matching(r"campaign|pipeline|sweep|solve|runner|generate")))
    if not scope:
        return _dim(None, [], "no solver-driving code identified")

    pool = idx.search(r"multiprocessing\.Pool|ProcessPoolExecutor|Pool\s*\(", scope)
    if pool:
        return _dim("process_pool", pool)
    sub = [h for h in idx.search(
        r"subprocess\.(?:run|Popen|check_call|check_output)", scope)
        if re.search(r"solve|campaign|worker|batch|sys\.executable", h["text"], re.I)]
    if sub:
        return _dim("subprocess_per_batch", sub)
    forked = idx.search(r"os\.fork|SpawnProcess|\bProcess\s*\(", scope)
    if forked:
        return _dim("forked_process", forked)
    return _dim("in_process_serial", idx.search(r"OFT_env\s*\(", scope),
                "solves run in the driving process, one after another")


def _mesh_route(idx: CodeIndex) -> dict[str, Any]:
    """The task forbids hand-built triangulations; did they use the OFT API?"""
    gs = idx.search(r"gs_Domain|define_region|add_(?:annulus|polygon|rectangle)"
                    r"|build_mesh|set_mesh")
    hand = idx.search(r"scipy\.spatial\.Delaunay|Delaunay\s*\(|triangulate\s*\(")
    if hand:
        return _dim("hand_built_triangulation", hand,
                    "explicitly forbidden by the task")
    if gs:
        return _dim("oft_gs_domain", gs)
    return _dim(None, [])


def _mask_rule(idx: CodeIndex) -> dict[str, Any]:
    """How "NaN outside the plasma" is decided — the defect that passed 9/9 gates.

    Answered from the predictor and the solver wrapper only. A run that masks
    by testing membership of the LCFS polygon has solved a different problem
    from one that reuses whichever pixels happened to be finite in training.
    """
    # Any production file that actually writes NaN into a field belongs in
    # scope: the masking helper is routinely in model.py, not "mask.py".
    scope = (idx.files_matching(r"predict")
             + idx.files_containing(r"OFT_env\s*\(")
             + idx.files_matching(r"mask|field|solver|psi")
             + idx.files_containing(r"np\.nan|np\.where\("))
    scope = list(dict.fromkeys(scope)) or list(idx.files)

    rules: list[tuple[str, str]] = [
        # Polygon containment, whatever the helper is called: agents ship
        # _points_in_poly, inside_lcfs_mask, boundary_mask.
        ("lcfs_polygon", r"contains_points|inside_\w*(?:polygon|lcfs)|matplotlib\.path"
                         r"|points?_in_poly|boundary_mask|winding_number"),
        ("psi_norm_threshold", r"psi_norm|psi_n\b|normalized_?flux\s*[<>]|>\s*1\.0\s*\)"),
        ("solver_native_nan", r"get_psi\(|psi\[.*\]\s*=\s*np\.nan.*solver|raw_nan"),
        ("training_valid_mask", r"valid_mask|finite_mask|isfinite\(.*train|mask_from_data"),
    ]
    found = []
    for name, pat in rules:
        hits = idx.search(pat, files=scope)
        if hits:
            found.append((name, hits))
    if not found:
        return _dim(None, [], "no masking rule identified")
    value = "+".join(n for n, _ in found)
    return _dim(value, [h for _, hits in found for h in hits[:1]])


def _grid_mapping(idx: CodeIndex) -> dict[str, Any]:
    """Triangular mesh -> the frozen 64x96 rectangle."""
    for name, pat in (
        ("solver_field_eval", r"get_field_eval|eval_field|get_psi\(.*R.*Z|sample_field"),
        ("scipy_interpolator", r"RegularGridInterpolator|LinearNDInterpolator|griddata"
                               r"|interp2d|CloughTocher"),
        ("manual_barycentric", r"barycentric|simplex|find_simplex"),
    ):
        hits = idx.search(pat)
        if hits:
            return _dim(name, hits)
    return _dim(None, [])


def _storage(idx: CodeIndex) -> dict[str, Any]:
    writers = [
        ("npz_per_solve", r"np\.savez(?:_compressed)?\s*\("),
        ("hdf5", r"h5py\.File|to_hdf|create_dataset"),
        ("npy", r"np\.save\s*\("),
        ("pickle", r"pickle\.dump|joblib\.dump|torch\.save"),
    ]
    found = [(n, idx.search(p)) for n, p in writers]
    found = [(n, h) for n, h in found if h]
    if not found:
        return _dim(None, [])
    # Index/manifest alongside the arrays is a real design difference: it is
    # what makes a campaign resumable and auditable.
    index = idx.search(r"index\.csv|solves_index|manifest|registry|catalog")
    value = "+".join(n for n, _ in found[:2])
    if index:
        value += "+index"
    return _dim(value, [h for _, hits in found for h in hits[:1]] + index[:1])


def _storage_validation(idx: CodeIndex) -> dict[str, Any]:
    """Task-mandated: a solve counts only after its stored file re-loads finite."""
    hits = []
    for rel in idx.files:
        for start, _end, name in idx._funcs.get(rel, []):
            body = idx.function_source(rel, name)
            if not body:
                continue
            loads = re.search(r"np\.load|h5py\.File|load\s*\(", body)
            checks = re.search(r"isfinite|finite_frac|np\.isnan|nan_fraction", body)
            if loads and checks:
                hits.append({"where": f"{rel}:{start}", "file": rel, "line": start,
                             "text": name, "function": name})
    if hits:
        return _dim("reload_and_check_finite", hits)
    weak = idx.search(r"finite_frac|stored_ok|validate_stored")
    return _dim("claimed_only" if weak else None, weak,
                "no function both re-loads and checks finiteness" if weak else "")


def _pilot_gate(idx: CodeIndex) -> dict[str, Any]:
    """Task-mandated: >=20 solves, stop below 50% success."""
    files = idx.files_matching(r"pilot") or idx.files_containing(r"\bpilot\b")
    if not files:
        return _dim(None, [])
    rate = idx.search(r"success_rate|rate\s*[<>]=?\s*0?\.5|0\.5\s*[<>]|success\s*/", files)
    return _dim("enforced" if rate else "run_without_threshold",
                (rate or idx.search(r"\bpilot\b", files)))


def _self_test(idx: CodeIndex) -> dict[str, Any]:
    """Task-mandated: re-run the documented CLI in a fresh process."""
    hits = [h for h in idx.search(r"predict\.py", everywhere=True)
            if re.search(r"subprocess|sys\.executable|check_call|run\(", h["text"], re.I)]
    if hits:
        return _dim("subprocess_reruns_predict", hits)
    mention = idx.search(r"predict\.py", everywhere=True)
    return _dim("imported_not_subprocessed" if mention else None, mention)


def _leakage_guard(idx: CodeIndex) -> dict[str, Any]:
    """Does the held-out test set stay out of the code that fits the model?

    Function-scoped on purpose. "The file that trains also mentions test" is
    true of any single-file pipeline and says nothing; "the function that
    calls .fit() also touches a test path" is a specific thing to go and read.
    A reference is a flag for review, never a proof of leakage — the task
    forbids the test set influencing training, selection OR acquisition, and
    only the first of those is visible this way.
    """
    fitters, touched = [], []
    for rel in idx.production:
        for start_line, _end, name in idx._funcs.get(rel, []):
            body = idx.function_source(rel, name)
            if not body or not re.search(
                    r"\.fit\(|\.backward\(|optimizer\.step|\.train\(\)"
                    r"|train_epoch|fit_model", body):
                continue
            fitters.append(f"{rel}:{start_line}")
            for i, line in enumerate(body.splitlines(), start=start_line):
                if re.search(r"(?:^|[^a-zA-Z])test\w*", line) and not re.search(
                        r"^\s*#|self[_-]?test|pytest|unittest", line):
                    touched.append({"where": f"{rel}:{i}", "file": rel, "line": i,
                                    "text": line.strip()[:160], "function": name})
                    break
    if not fitters:
        return _dim(None, [], "no model-fitting function identified")
    if touched:
        return _dim("test_referenced_in_fitting_function", touched,
                    f"{len(touched)} of {len(fitters)} fitting functions — read the lines")
    return _dim("test_absent_from_fitting_functions",
                [{"where": f} for f in fitters[:MAX_EVIDENCE]],
                f"{len(fitters)} fitting function(s) checked")


def _code_shape(idx: CodeIndex) -> dict[str, Any]:
    """Monolith or modules — how the solution is organised at all."""
    n_files = len(idx.production)
    sloc = sum(len([ln for ln in idx.files[rel].splitlines()
                    if ln.strip() and not ln.strip().startswith("#")])
               for rel in idx.production)
    biggest = max(((len(idx.files[rel].splitlines()), rel) for rel in idx.production),
                  default=(0, None))
    if n_files <= 2:
        value = "single_script"
    elif biggest[0] > 0.6 * max(sloc, 1):
        value = "one_dominant_module"
    else:
        value = f"{n_files}_modules"
    scaffolding = len(idx.files) - n_files
    return _dim(value, [{"where": f"{biggest[1]}:1"}] if biggest[1] else [],
                f"{sloc} sloc across {n_files} pipeline files"
                + (f" (+{scaffolding} smoke/test/repro scripts)" if scaffolding else ""))


def _entry_point(idx: CodeIndex) -> dict[str, Any]:
    mains = [rel for rel in idx.production
             if re.search(r'__name__\s*==\s*[\'"]__main__', idx.files[rel])]
    runner = [r for r in mains if re.search(r"run_pipeline|main|pipeline|campaign", r, re.I)]
    value = (f"{len(mains)}_runnable_scripts" if len(mains) > 1 else
             ("single_entry_script" if mains else None))
    return _dim(value, [{"where": f"{r}:1"} for r in (runner or mains)[:MAX_EVIDENCE]])


# The rows of the cross-comparison, in the order the pipeline executes.
DIMENSIONS: dict[str, Any] = {
    "oft_env_strategy": _oft_strategy,
    "solve_isolation": _solve_isolation,
    "mesh_route": _mesh_route,
    "grid_mapping": _grid_mapping,
    "mask_rule": _mask_rule,
    "storage": _storage,
    "storage_validation": _storage_validation,
    "pilot_gate": _pilot_gate,
    "leakage_guard": _leakage_guard,
    "self_test": _self_test,
    "code_shape": _code_shape,
    "entry_point": _entry_point,
}

# What each row is asking, for the report's benefit — a reader should not
# have to open the source to know what "per_worker_process" is answering.
DIMENSION_QUESTIONS: dict[str, str] = {
    "oft_env_strategy": "OFT allows one OFT_env per process, ever. Who owns it?",
    "solve_isolation": "What insulates one solve from the next?",
    "mesh_route": "Meshed through the OFT API, or hand-built (which the task forbids)?",
    "grid_mapping": "How does the triangular mesh reach the frozen 64x96 grid?",
    "mask_rule": "How is 'no plasma here' decided when writing NaN?",
    "storage": "What is a solved sample on disk, and is it indexed?",
    "storage_validation": "Is a solve counted only after its file re-loads finite?",
    "pilot_gate": "Was the mandated pilot run, and was its 50% threshold enforced?",
    "leakage_guard": "Does the test set stay out of the functions that fit the model?",
    "self_test": "Was the documented predict.py CLI re-run in a fresh process?",
    "code_shape": "One script or a module tree?",
    "entry_point": "How would a stranger run it?",
}


# Every value the detectors above can emit, defined. A code in a report cell
# that a reader has to guess at is not a measurement, it is a rumour — so the
# definitions live next to the detector that emits them and are rendered as
# hover text wherever the code appears.
VALUE_GLOSSARY: dict[str, str] = {
    # --- oft_env_strategy ---
    "singleton_reused": (
        "One OFT_env is built once and handed out on every call — a cached "
        "module global, a class guarding construction, or an lru_cache. "
        "Honours OFT's one-env-per-process limit in the simplest way, at the "
        "cost of keeping every solve in one process: a solver crash takes the "
        "campaign with it."),
    "per_worker_process": (
        "Each worker process builds its own OFT_env. The only arrangement "
        "that lets the campaign use more than one core, since the limit is "
        "per process — and it contains a crash to one worker. Costs a fresh "
        "env (and mesh) per worker."),
    "per_solve": (
        "An OFT_env is constructed inside the per-solve function. Directly "
        "against the documented limit if those solves share a process; "
        "usually a latent crash rather than a design."),
    "module_level_once": (
        "OFT_env is constructed at import or in one top-level block, with no "
        "guard. Works while the module is imported once, with nothing "
        "stopping a second construction."),
    # --- solve_isolation ---
    "in_process_serial": (
        "Solves run one after another inside the driving process. Simplest, "
        "and the slowest: no parallelism, and one bad solve can poison the "
        "shared solver state."),
    "process_pool": (
        "multiprocessing.Pool / ProcessPoolExecutor fans solves across "
        "processes. Parallel, and each solve is insulated — a segfault kills "
        "a worker, not the campaign."),
    "subprocess_per_batch": (
        "The driver shells out (usually to its own script) to run a batch. "
        "Insulates solver state between batches and survives a hard crash; "
        "pays process start-up and serialises through files."),
    "forked_process": (
        "Solves run in explicitly forked/spawned processes without a pool — "
        "the isolation of a pool, managed by hand."),
    # --- mesh_route ---
    "oft_gs_domain": (
        "Meshed through OFT's own API (gs_Domain / define_region / build_mesh) "
        "as the worked examples do. What the task's meshing milestone asks "
        "for, and what TokaMaker's topology checks accept."),
    "hand_built_triangulation": (
        "A mesh built outside OFT (e.g. scipy.spatial.Delaunay) and handed to "
        "the solver. The task forbids this explicitly: TokaMaker validates "
        "mesh topology and rejects hand-built triangulations."),
    # --- grid_mapping ---
    "solver_field_eval": (
        "psi is read on the frozen 64x96 grid through the solver's own finite-"
        "element interpolator (get_field_eval). Evaluates the FE basis "
        "directly, so no second interpolation error is introduced."),
    "scipy_interpolator": (
        "The solver's nodal values are re-interpolated onto the grid with "
        "scipy (RegularGridInterpolator / griddata / CloughTocher). Adds an "
        "interpolation error on top of the solve, and depends on how the "
        "nodes were sampled first."),
    "manual_barycentric": (
        "The grid point is located in the mesh by hand (find_simplex / "
        "barycentric weights) and psi blended from the triangle's vertices."),
    # --- mask_rule ---
    "lcfs_polygon": (
        "A grid point is inside the plasma iff it lies within the LCFS "
        "polygon — point-in-polygon against the D-shape boundary, whatever "
        "the helper is called (contains_points, _points_in_poly, "
        "inside_lcfs_mask). Geometric, per sample, and independent of the "
        "surrogate's own output."),
    "psi_norm_threshold": (
        "Inside/outside is decided by normalised flux (psi_n <= 1). Uses the "
        "solve's own notion of the boundary rather than the requested shape; "
        "sensitive to how psi_n is normalised."),
    "solver_native_nan": (
        "Whatever the solver itself leaves undefined outside the plasma is "
        "kept as NaN. No separate mask to get wrong — and no mask at "
        "prediction time either, unless one is reconstructed."),
    "training_valid_mask": (
        "The predictor masks with a valid-pixel mask derived from the "
        "training data (e.g. pixels finite in training). Cheap and stable, "
        "but the mask cannot adapt to a geometry unlike those seen in "
        "training — the same family of defect as the run that passed 9/9 "
        "gates while predicting plasma in vacuum."),
    # --- storage ---
    "npz_per_solve": "One compressed .npz per solved sample.",
    "hdf5": (
        "One HDF5 file holding many samples. Scales best and carries "
        "attributes/provenance, at the price of concurrent-write care."),
    "npy": "Raw .npy arrays, typically one per field per sample.",
    "pickle": (
        "Python-pickled objects (pickle/joblib/torch.save) — usually the "
        "model artifact rather than the dataset."),
    "index": (
        "An index/manifest (CSV or JSONL) sits beside the arrays, listing "
        "every sample and its status. What makes a campaign resumable and "
        "auditable rather than a directory to be re-scanned."),
    # --- storage_validation ---
    "reload_and_check_finite": (
        "A function both re-loads a written artifact and checks its finite "
        "fraction — the task's storage-validation gate, actually implemented. "
        "This is the check that catches an all-NaN grid stored as a success."),
    "claimed_only": (
        "The workspace talks about stored-file validation (finite_frac, "
        "stored_ok) but no function both loads a file and checks finiteness. "
        "The gate is reported, not performed."),
    # --- pilot_gate ---
    "enforced": (
        "A pilot runs and its success rate is compared against a threshold "
        "before the campaign proceeds — the task's 50% pilot gate."),
    "run_without_threshold": (
        "A pilot runs, but nothing compares its success rate to a threshold: "
        "the campaign starts regardless of what the pilot found."),
    # --- leakage_guard ---
    "test_absent_from_fitting_functions": (
        "No function that calls .fit()/backward()/optimizer.step() references "
        "a test path or array. Necessary, not sufficient: the task also "
        "forbids the test set influencing model SELECTION and ACQUISITION, "
        "which this check cannot see."),
    "test_referenced_in_fitting_function": (
        "A function that fits the model also touches something named test. A "
        "flag for review, not a verdict — read the cited line; it may be a "
        "legitimate final-evaluation call sharing a function."),
    # --- self_test ---
    "subprocess_reruns_predict": (
        "Something in the workspace invokes predict.py as a subprocess with "
        "the contract CLI — the task's deliverable self-test, in a fresh "
        "process, which is the only way to catch an import-order or "
        "global-state dependency."),
    "imported_not_subprocessed": (
        "predict.py is referenced but only imported or described, never run "
        "as the documented command. Code that works in the session's process "
        "and fails in a clean one is exactly what this misses."),
    # --- code_shape / entry_point ---
    "single_script": "The whole pipeline is one or two files.",
    "one_dominant_module": (
        "One file holds most of the pipeline's lines, with small helpers "
        "around it."),
    "single_entry_script": "Exactly one file is runnable as a program.",
}

# Longer framing for each row of the cross comparison: what the dimension is,
# and why a difference in it matters.
DIMENSION_NOTES: dict[str, str] = {
    "oft_env_strategy": (
        "TokaMaker allows exactly one OFT_env per Python process, for the "
        "life of that process. Everything about how a campaign is executed "
        "follows from how the agent chose to live with that."),
    "solve_isolation": (
        "Whether solves share a process decides both throughput and blast "
        "radius: in one process a single solver crash ends the campaign."),
    "mesh_route": (
        "The task mandates reproducing the OFT fixed-boundary example's "
        "meshing workflow and forbids hand-built triangulations, which "
        "TokaMaker's topology validation rejects."),
    "grid_mapping": (
        "Ground truth is defined on a frozen 64x96 R/Z grid; the solve is on "
        "a triangular mesh. How that gap is crossed is a source of error "
        "entirely separate from the surrogate."),
    "mask_rule": (
        "psi is undefined outside the plasma and the contract requires NaN "
        "there. Getting this wrong produced the headline failure of this "
        "benchmark: a run that passed every gate while emitting a "
        "full-magnitude field in vacuum."),
    "storage": (
        "What a solved sample is on disk, and whether an index makes the "
        "campaign resumable and auditable."),
    "storage_validation": (
        "The task counts a solve as successful only after its stored artifact "
        "re-loads with finite values — added after an agent stored all-NaN "
        "grids and reported them as successes."),
    "pilot_gate": (
        "The task requires a >=20-solve pilot and a stop below 50% success, "
        "so that a broken solver is found before 300 solves are spent on it."),
    "leakage_guard": (
        "The held-out test set must never influence training, selection or "
        "acquisition. Only the training half of that is visible in code."),
    "self_test": (
        "The task requires re-running each deliverable's documented command "
        "in a fresh process before declaring done."),
    "code_shape": (
        "How the solution is organised at all — one script or a module tree. "
        "Not a quality judgement on its own; read it next to the score."),
    "entry_point": (
        "How a stranger would run it, which the task requires to be one "
        "documented command."),
}


def analyse_solution_shape(workspace: Path) -> dict[str, Any]:
    """Every dimension of the approach, each with its code evidence."""
    workspace = Path(workspace)
    try:
        idx = CodeIndex(workspace)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "dimensions": {}}
    if not idx.files:
        return {"dimensions": {}, "n_files": 0,
                "note": "no agent-authored Python found"}

    dims: dict[str, Any] = {}
    for name, fn in DIMENSIONS.items():
        try:
            dims[name] = fn(idx)
        except Exception as exc:  # noqa: BLE001 — one bad dimension, not a lost run
            dims[name] = {"value": None, "evidence": [],
                          "note": f"{type(exc).__name__}: {exc}"}
    return {
        "dimensions": dims,
        "n_files": len(idx.files),
        "n_pipeline_files": len(idx.production),
        # A one-line fingerprint of the approach, for grouping replicates.
        "shape_signature": " | ".join(
            f"{k}={dims[k]['value']}" for k in
            ("oft_env_strategy", "solve_isolation", "mask_rule", "grid_mapping")
            if dims.get(k, {}).get("value")),
    }


def cross_compare(records: dict[str, dict]) -> list[dict[str, Any]]:
    """Transpose: one row per dimension, one column per condition.

    A cross-comparison is a transposition — the question is "how did they
    each answer THIS", and that only reads as a comparison when the answers
    sit on one line.
    """
    rows = []
    for dim in DIMENSIONS:
        row: dict[str, Any] = {"dimension": dim,
                               "question": DIMENSION_QUESTIONS.get(dim, "")}
        values = []
        for label, rec in records.items():
            d = (rec.get("dimensions") or {}).get(dim) or {}
            row[label] = d.get("value") or "-"
            values.append(d.get("value"))
        known = [v for v in values if v]
        row["n_distinct"] = len(set(known))
        # The rows worth reading first: where the agents disagreed.
        row["agreement"] = ("all_same" if known and len(set(known)) == 1
                            else ("all_differ" if len(set(known)) == len(known) and known
                                  else "mixed"))
        rows.append(row)
    return rows


__all__ = [
    "DIMENSIONS",
    "DIMENSION_NOTES",
    "DIMENSION_QUESTIONS",
    "VALUE_GLOSSARY",
    "CodeIndex",
    "analyse_solution_shape",
    "cross_compare",
]
