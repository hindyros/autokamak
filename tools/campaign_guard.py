#!/usr/bin/env python
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Guards that stand between a campaign and a wasted budget.

A replicated campaign is the most expensive thing this repo does: tens of
paid agent sessions, hours of wall clock, and nothing recoverable if it goes
wrong in the wrong way. The failures that actually cost money are not exotic:

  * a crash or Ctrl-C at hour 18, and the re-run repeats every finished cell
    because ``bench run`` always mints a new run_id (``completed``);
  * a harness that should not be in the campaign at all — this paper is
    pinned to OpenAI models, and ``claude_sdk`` is Anthropic by construction
    (``preflight``);
  * a missing key, a logged-out CLI or a full disk discovered at 2am, after
    the cells have already burned their timeout (``preflight``);
  * spend running away while nobody is watching (``spend``, used by the
    driver to stop the campaign at a ceiling).

Each subcommand is exit-code-driven so ``tools/run_campaign.sh`` can use it
as a gate.

    python tools/campaign_guard.py preflight --harnesses "ursa dspy pi cursor" \
        --task benchmarks/tasks/L3_mini_v3.yaml [--min-disk-gb 10]
    python tools/campaign_guard.py completed --tag t --condition L3-pi --rep 2
    python tools/campaign_guard.py spend --tag t [--budget 150] [--harness ursa]
    python tools/campaign_guard.py ratelimits [--tag t] [--parallel 3]
    python tools/campaign_guard.py forecast --harnesses "ursa dspy pi cursor" --reps 5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO_ROOT / "experiments"
RATE_LIMITS = REPO_ROOT / "benchmarks" / "assets" / "rate_limits.json"

# Blended $/1M tokens at the pinned model's rates, used ONLY to invert a
# dollar figure into a token estimate for substrates that report cost but
# not usage (URSA). Derived from the observed ~20:1 input:output ratio at
# $1.75 in / $14.00 out.
BLENDED_USD_PER_MTOK = 2.33

# Substrate -> (env var that must be set, CLI that must be on PATH).
HARNESS_REQUIREMENTS: dict[str, tuple[str | None, str | None]] = {
    "ursa": ("OPENAI_API_KEY", None),
    "dspy": ("OPENAI_API_KEY", None),
    "pi": ("OPENAI_API_KEY", "pi"),
    "cursor": (None, "cursor-agent"),          # CURSOR_API_KEY or a CLI login
    "claude_sdk": ("ANTHROPIC_API_KEY", "claude"),
    "echo": (None, None),
}

# Harnesses that cannot honour an OpenAI model pin. Running one silently
# confounds the harness axis with the model axis AND spends on another
# provider, so the driver refuses unless told otherwise.
NON_OPENAI = {"claude_sdk"}


def load_rate_limits(model: str) -> dict:
    """This org's limits for a model, from the committed dashboard record.

    TPD is the reason this file exists: TPM and RPM come back in response
    headers, the daily cap does not, and at campaign scale the daily cap is
    the one that bites.
    """
    try:
        data = json.loads(RATE_LIMITS.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    models = data.get("models") or {}
    for name, limits in models.items():
        if name in model or model in name:
            return {**limits, "_as_of": data.get("as_of"), "_model": name}
    return {}


def _runs(tag_dir: Path):
    for rp in sorted(tag_dir.glob("*/*/result.json")):
        try:
            yield rp, json.loads(rp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue


def cmd_completed(args) -> int:
    """Exit 0 when this (condition, replicate) already finished. For --resume."""
    tag_dir = EXPERIMENTS / args.tag
    if not tag_dir.is_dir():
        return 1
    for _rp, r in _runs(tag_dir):
        if r.get("condition") != args.condition:
            continue
        if args.rep is not None and r.get("replicate") != args.rep:
            continue
        # A timed-out or errored cell is NOT done: re-running it is the point
        # of a resume. Only a completed run counts.
        if r.get("status") == "completed":
            print(f"{args.condition} rep {args.rep}: already completed "
                  f"({_rp.parent.name})")
            return 0
    return 1


def measured_spend(tag: str, harness: Optional[str] = None
                   ) -> tuple[float, list[str]]:
    """Dollars spent under a tag, best-effort, with what could not be priced.

    Self-reported cost first; then tools/cost_report.py's derived figures
    (cursor reports tokens, not dollars); then URSA's own per-call accounting
    for runs the watchdog killed before they could write a result.json — the
    aborted n=10 attempt hid $12.31 of real spend that way.
    """
    tag_dir = EXPERIMENTS / tag
    total, unpriced = 0.0, []
    if not tag_dir.is_dir():
        return 0.0, ["no such tag dir"]

    def _match(condition: Optional[str]) -> bool:
        # Conditions are "<level>-<harness>"; a per-harness ceiling spans
        # levels, because it is the SUBSTRATE that runs away, not the level.
        return harness is None or str(condition).partition("-")[2] == harness

    priced_runs = set()
    for rp, r in _runs(tag_dir):
        run_key = (r.get("condition"), rp.parent.name)
        if not _match(run_key[0]):
            continue
        c = r.get("cost_usd")
        if isinstance(c, (int, float)):
            total += float(c)
            priced_runs.add(run_key)
        else:
            unpriced.append(f"{run_key[0]}/{run_key[1]}")

    csv_path = tag_dir / "cost_report.csv"
    if csv_path.is_file():
        import csv as _csv

        try:
            with csv_path.open(newline="") as fh:
                for row in _csv.DictReader(fh):
                    key = (row.get("condition"), row.get("run_id"))
                    if key in priced_runs or not _match(key[0]):
                        continue
                    try:
                        total += float(row["cost_usd"])
                        priced_runs.add(key)
                        if f"{key[0]}/{key[1]}" in unpriced:
                            unpriced.remove(f"{key[0]}/{key[1]}")
                    except (KeyError, TypeError, ValueError):
                        continue
        except OSError:
            pass

    # Killed runs: no result.json, but URSA wrote its own accounting.
    for metrics in tag_dir.glob("*/*/workspace/ursa_metrics/*.json"):
        key = (metrics.parents[2].parent.name, metrics.parents[2].name)
        if key in priced_runs or not _match(key[0]):
            continue
        try:
            data = json.loads(metrics.read_text(encoding="utf-8"))
            c = (data.get("costs") or {}).get("total_usd")
            if isinstance(c, (int, float)):
                total += float(c)
                priced_runs.add(key)
        except Exception:  # noqa: BLE001
            continue
    return total, unpriced


# Worst observed cost and duration per substrate, from the archived v3
# shakedown (experiments/matrix-v3-n10-20260917 + the aborted n=10 attempt).
# Used only to FORECAST exposure before spending; measured spend always
# comes from the runs themselves.
# "tokens" is per run, charged to THIS key: measured where the substrate
# reports usage, inferred from spend where it does not (ursa), and zero for
# cursor, which consumes Cursor's quota rather than ours.
OBSERVED_WORST: dict[str, dict[str, float]] = {
    # Recalibrated on the 2026-09-19 pilot, which is the first run where all
    # four substrates were measured under one configuration. ursa is the
    # correction that mattered: it burns $0.179/min (it timed out at 20.6 min
    # having spent $3.68), 57% dearer per minute than the earlier estimate
    # taken from the aborted long run.
    "ursa": {"usd": 16.1, "minutes": 90.0, "tokens": 7.4e6},
    "dspy": {"usd": 3.0, "minutes": 28.0, "tokens": 2.93e6},
    "pi": {"usd": 2.0, "minutes": 15.0, "tokens": 5.68e6},
    "cursor": {"usd": 1.7, "minutes": 17.0, "tokens": 0.0},
    "claude_sdk": {"usd": 6.0, "minutes": 40.0, "tokens": 0.0},
    "echo": {"usd": 0.0, "minutes": 1.0, "tokens": 0.0},
}


def cmd_forecast(args) -> int:
    """Worst-case exposure before a single call is made.

    A budget ceiling only tells you where the campaign STOPS; this tells you
    what it can cost if every cell runs to its worst observed length. The
    per-harness rows are what matter: one slow substrate can eat a shared
    ceiling and starve the other three, which is why the driver takes
    per-harness caps as well as a global one.
    """
    def _cap(spec: str, name: str):
        for part in (spec or "").split(","):
            k, _, v = part.partition("=")
            if k.strip() == name:
                try:
                    return float(v)
                except ValueError:
                    return None
        return None

    total_usd = total_min = total_tok = 0.0
    rows = []
    n_levels = max(1, len(args.levels.split()))
    for spec in args.harnesses.split():
        name, _, cap = spec.partition(":")
        reps = int(cap) if cap.isdigit() else args.reps
        w = dict(OBSERVED_WORST.get(name, {"usd": 5.0, "minutes": 60.0,
                                           "tokens": 3e6}))
        # A shorter per-cell cap truncates the worst case: spend tracks time
        # for a substrate that runs until it is stopped.
        tcap = _cap(args.harness_timeout, name)
        if tcap and tcap / 60 < w["minutes"]:
            scale = (tcap / 60) / w["minutes"]
            w["usd"] *= scale
            w["tokens"] = w.get("tokens", 0.0) * scale
            w["minutes"] = tcap / 60
        # An explicit ring-fence is a hard ceiling, whatever the cells cost.
        bcap = _cap(args.harness_budget, name)
        cells = reps * n_levels
        usd = cells * w["usd"]
        if bcap is not None:
            usd = min(usd, bcap)
        minutes = cells * w["minutes"]
        tokens = cells * w.get("tokens", 0.0)
        total_usd += usd
        total_min += minutes
        total_tok += tokens
        rows.append((name, cells, w["usd"], usd, w["minutes"], minutes, tokens))

    print(f"{'harness':<12}{'cells':>6}{'$/cell':>9}{'$ worst':>10}"
          f"{'min/cell':>10}{'h serial':>10}{'Mtok on key':>13}")
    for name, cells, per, usd, per_min, minutes, tokens in sorted(
            rows, key=lambda r: -r[3]):
        print(f"{name:<12}{cells:>6}{per:>9.2f}{usd:>10.2f}"
              f"{per_min:>10.0f}{minutes / 60:>10.1f}{tokens / 1e6:>13.1f}")
    print(f"{'TOTAL':<12}{sum(r[1] for r in rows):>6}{'':>9}{total_usd:>10.2f}"
          f"{'':>10}{total_min / 60:>10.1f}{total_tok / 1e6:>13.1f}")

    # The daily cap is the campaign-scale limit: TPM bounds a burst, TPD
    # bounds the day, and hitting it stops every substrate at once.
    limits = load_rate_limits(args.model)
    tpd = args.tpd or limits.get("tpd")
    if tpd and total_tok:
        pct = total_tok / tpd
        print(f"\nDaily tokens: {total_tok / 1e6:.0f}M of the {tpd / 1e6:.0f}M TPD "
              f"cap ({pct:.0%}) if the whole campaign runs inside one day.")
        if pct >= 0.85:
            print("  BLOCKER-ish: split it across days (--resume continues "
                  "where it stopped) — the daily wall stops every substrate "
                  "at once, mid-run.")
        elif pct >= 0.5:
            print("  Fits, but a re-run or a second arm on the same day would "
                  "not. Cached input is counted here; if it turns out not to "
                  "consume quota this figure roughly halves.")
        print("  (cursor excluded: it consumes Cursor's quota, not this key's)")
    print(f"\nWall clock at --parallel {args.parallel}: "
          f"~{total_min / 60 / max(1, args.parallel):.1f} h worst case "
          f"(cells are not evenly sized, so treat it as an upper bound).")
    if rows:
        worst = max(rows, key=lambda r: r[3])
        share = worst[3] / total_usd * 100 if total_usd else 0
        print(f"Largest exposure: {worst[0]} at ${worst[3]:.2f} "
              f"({share:.0f}% of the worst case). Ring-fence it with "
              f"--harness-budget \"{worst[0]}=<usd>\" so it cannot starve "
              f"the others.")
    return 0


def _ratelimit_headers(model: str) -> tuple[dict, Optional[str]]:
    """One minimal completion, read for its rate-limit headers.

    Costs a fraction of a cent and is the only way to know this key's ACTUAL
    limits: they are per-account, per-model and change with tier, so any
    number written into a repo is a guess with a shelf life.
    """
    import httpx

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return {}, "OPENAI_API_KEY is not set"
    try:
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model,
                  "messages": [{"role": "user", "content": "hi"}],
                  "max_completion_tokens": 1},
            timeout=60.0,
        )
    except Exception as exc:  # noqa: BLE001
        return {}, f"probe failed: {type(exc).__name__}: {exc}"
    headers = {k.lower(): v for k, v in r.headers.items()
               if k.lower().startswith("x-ratelimit")}
    if r.status_code >= 400 and not headers:
        return {}, f"probe returned HTTP {r.status_code}: {r.text[:200]}"
    return headers, None


def _parse_limit(value: Optional[str]) -> Optional[float]:
    """OpenAI writes these as 10000, 30000000, or 1.5k / 2m."""
    if not value:
        return None
    v = value.strip().lower()
    mult = 1.0
    if v.endswith("k"):
        mult, v = 1e3, v[:-1]
    elif v.endswith("m"):
        mult, v = 1e6, v[:-1]
    try:
        return float(v) * mult
    except ValueError:
        return None


def observed_session_rates(tag: Optional[str]) -> list[tuple[str, float, float]]:
    """(condition, tokens-per-minute, requests-per-minute) per archived run.

    Measured, not assumed: a campaign's rate-limit risk is entirely about how
    fast ONE agent session consumes tokens, and that varies ~4x across these
    substrates (a cached-heavy cursor session burns far more TPM than a dspy
    one at the same dollar cost).
    """
    import csv as _csv

    out: list[tuple[str, float, float]] = []
    if not tag:
        return out
    csv_path = EXPERIMENTS / tag / "cost_report.csv"
    if not csv_path.is_file():
        return out
    try:
        with csv_path.open(newline="") as fh:
            for row in _csv.DictReader(fh):
                try:
                    minutes = float(row.get("wall_min") or 0)
                    if minutes <= 0:
                        continue
                    tokens = sum(float(row.get(k) or 0)
                                 for k in ("tok_in", "tok_out", "tok_cache"))
                    turns = float(row.get("turns") or 0)
                    out.append((row.get("condition", "?"), tokens / minutes,
                                turns / minutes))
                except (TypeError, ValueError):
                    continue
    except OSError:
        pass
    return out


def _analyze_tokens(run_dir: Path) -> dict:
    """Tokens for one run, reusing cost_report's extraction."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from cost_report import DEFAULT_PRICES, analyze_run  # type: ignore

    prices = dict(DEFAULT_PRICES)
    committed = REPO_ROOT / "benchmarks" / "assets" / "prices.json"
    if committed.is_file():
        try:
            data = json.loads(committed.read_text(encoding="utf-8"))
            prices.update({k: v for k, v in (data.get("prices") or {}).items()
                           if isinstance(v, dict) and v.get("in") is not None})
        except Exception:  # noqa: BLE001
            pass
    try:
        return analyze_run(run_dir, prices)
    except Exception:  # noqa: BLE001
        return {}


def tokens_used(tag: str, *, on_key_only: bool = True,
                since_hours: Optional[float] = None) -> tuple[float, dict]:
    """Tokens charged to THIS key under a tag, with a per-substrate split.

    Cached input is counted: whether it consumes quota is not documented on
    the dashboard, and assuming it does is the direction that cannot cause
    an unplanned stop. cursor is excluded by default — it reaches the model
    through Cursor's backend, so its (very large, ~95% cached) usage never
    touches this key.

    URSA reports dollars but no usage, so its tokens are INFERRED from spend
    at the pinned model's blended rate and marked as such.
    """
    import datetime as _dt

    tag_dir = EXPERIMENTS / tag
    by_harness: dict[str, dict] = {}
    if not tag_dir.is_dir():
        return 0.0, by_harness

    cutoff = None
    if since_hours:
        cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=since_hours)

    total = 0.0
    for run_dir in sorted(tag_dir.glob("*/*")):
        if not run_dir.is_dir():
            continue
        condition = run_dir.parent.name
        harness = condition.partition("-")[2]
        if on_key_only and harness == "cursor":
            continue
        if cutoff is not None:
            try:
                stamp = _dt.datetime.fromtimestamp(run_dir.stat().st_mtime,
                                                   _dt.timezone.utc)
                if stamp < cutoff:
                    continue
            except OSError:
                pass
        row = _analyze_tokens(run_dir)
        toks = sum(float(row.get(k) or 0)
                   for k in ("tok_in", "tok_out", "tok_cache"))
        inferred = False
        if not toks:
            usd = row.get("cost_usd")
            if not isinstance(usd, (int, float)) or not usd:
                # URSA reports neither usage nor (when killed) a result.json;
                # its own per-call accounting is the only trace left.
                usd = 0.0
                for metrics in run_dir.glob("workspace/ursa_metrics/*.json"):
                    try:
                        data = json.loads(metrics.read_text(encoding="utf-8"))
                        usd += float((data.get("costs") or {}).get("total_usd") or 0)
                    except Exception:  # noqa: BLE001
                        continue
            if usd:
                toks = float(usd) / BLENDED_USD_PER_MTOK * 1e6
                inferred = True
        if not toks:
            continue
        slot = by_harness.setdefault(harness, {"tokens": 0.0, "runs": 0,
                                               "inferred": False})
        slot["tokens"] += toks
        slot["runs"] += 1
        slot["inferred"] = slot["inferred"] or inferred
        total += toks
    return total, by_harness


def cmd_tokens(args) -> int:
    """Tokens charged to this key under a tag, against the daily cap."""
    total, by_harness = tokens_used(args.tag, since_hours=args.since_hours)
    limits = load_rate_limits(args.model)
    tpd = args.tpd or limits.get("tpd")

    print(f"{total:.0f}")
    for h, d in sorted(by_harness.items(), key=lambda kv: -kv[1]["tokens"]):
        note = "  (inferred from spend; substrate reports no usage)" \
            if d["inferred"] else ""
        print(f"  {h:<12} {d['tokens']:>14,.0f} over {d['runs']:>2} run(s){note}",
              file=sys.stderr)
    if tpd:
        pct = total / tpd * 100
        print(f"  {'':<12} {pct:>13.1f}% of the {tpd:,.0f} TPD cap",
              file=sys.stderr)
        if total >= tpd * args.headroom:
            print(f"[guard] daily tokens {total:,.0f} have reached "
                  f"{args.headroom:.0%} of the {tpd:,.0f} TPD cap",
                  file=sys.stderr)
            return 2
    return 0


def cmd_ratelimits(args) -> int:
    """This key's real limits, against this campaign's measured appetite."""
    headers, err = _ratelimit_headers(args.model)
    if err:
        print(f"[ratelimits] {err}", file=sys.stderr)
        if args.tpm is None:
            print("[ratelimits] pass --tpm/--rpm to reason without a probe",
                  file=sys.stderr)
            return 1
    for k in sorted(headers):
        print(f"[ratelimits] {k}: {headers[k]}")

    limits = load_rate_limits(args.model)
    tpm = args.tpm or _parse_limit(headers.get("x-ratelimit-limit-tokens")) \
        or limits.get("tpm")
    rpm = args.rpm or _parse_limit(headers.get("x-ratelimit-limit-requests")) \
        or limits.get("rpm")
    # Not in any header: the daily cap is only knowable from the dashboard.
    tpd = args.tpd or limits.get("tpd")
    if tpm:
        print(f"[ratelimits] token limit: {tpm:,.0f} TPM")
    if rpm:
        print(f"[ratelimits] request limit: {rpm:,.0f} RPM")
    if tpd:
        print(f"[ratelimits] daily limit: {tpd:,.0f} TPD "
              f"(from benchmarks/assets/rate_limits.json, as_of "
              f"{limits.get('_as_of', '?')} — headers do not carry it)")

    rates = observed_session_rates(args.tag)
    if not rates:
        print("[ratelimits] no measured session rates "
              "(run tools/cost_report.py --tag <tag> first) — cannot size "
              "--parallel from evidence")
        return 0

    # cursor-agent calls the model through Cursor's own backend, so its
    # tokens are billed and rate-limited THERE, not against this key. Its
    # rate is still printed (it is a real load, on someone's quota) but it
    # must not size this key's parallelism.
    on_key = [r for r in rates if "cursor" not in r[0]]
    sizing = on_key or rates
    peak_tpm = max(r[1] for r in sizing)
    peak_rpm = max(r[2] for r in sizing)
    print("\n[ratelimits] measured per-session consumption "
          f"(n={len(rates)} archived runs):")
    for cond, t, rq in sorted(rates, key=lambda x: -x[1]):
        note = "  (billed via Cursor, not this key)" if "cursor" in cond else ""
        print(f"    {cond:<12} {t:>10,.0f} tok/min" +
              (f"  {rq:>6.1f} req/min" if rq else "") + note)

    if tpm:
        # Headroom factor: agent sessions are bursty — a session averaging
        # 300k TPM does not spread it evenly, and a 429 inside a harness is
        # retried at best and fatal at worst.
        safe = int(tpm * args.headroom / peak_tpm) if peak_tpm else 0
        print(f"\n[ratelimits] worst-case session burns {peak_tpm:,.0f} tok/min; "
              f"at {args.headroom:.0%} headroom that supports "
              f"--parallel {max(1, safe)}")
        if args.parallel and args.parallel > max(1, safe):
            print(f"[ratelimits] WARNING: --parallel {args.parallel} exceeds "
                  f"that. Expect 429s; substrates differ in whether they "
                  f"retry or fail the run.", file=sys.stderr)
            return 2
    if tpd:
        print(f"\n[ratelimits] the DAILY cap ({tpd:,.0f} TPD) is the one that "
              f"bites at campaign scale — check a plan against it with "
              f"`campaign_guard forecast`.")

    if rpm and peak_rpm and args.parallel:
        if peak_rpm * args.parallel > rpm * args.headroom:
            print(f"[ratelimits] WARNING: {args.parallel} x {peak_rpm:.0f} "
                  f"req/min approaches the {rpm:,.0f} RPM limit",
                  file=sys.stderr)
            return 2
    return 0


def cmd_reconcile(args) -> int:
    """Give killed cells a result.json so they survive into the analysis.

    A cell the driver's watchdog kills never writes one: the Python process
    dies mid-run. The run directory and its workspace remain, but every
    reporting path in this repo globs for ``result.json``, so the cell
    silently vanishes — the campaign's denominator shrinks and a substrate
    that always times out looks like a substrate that was never run. That is
    the difference between "ursa completed 0 of 8" and "ursa is absent from
    the table".

    The stub records what is knowable from the filesystem and nothing more;
    it is marked so it can never be mistaken for a measured run.
    """
    tag_dir = EXPERIMENTS / args.tag
    if not tag_dir.is_dir():
        print(f"no such tag dir: {tag_dir}", file=sys.stderr)
        return 1

    written = 0
    for run_dir in sorted(tag_dir.glob("*/*")):
        if not run_dir.is_dir() or (run_dir / "result.json").is_file():
            continue
        if not (run_dir / "workspace").is_dir() and not (run_dir / "trace.json").is_file():
            continue
        condition = run_dir.parent.name
        payload = {
            "condition": condition,
            "run_id": run_dir.name,
            "harness": condition.partition("-")[2],
            "status": "killed",
            "error": ("no result.json: the process was killed before it could "
                      "write one (driver watchdog, OOM, or an operator stop)"),
            "reconciled_by": "tools/campaign_guard.py reconcile",
            "measured": False,
        }
        trace_path = run_dir / "trace.json"
        if trace_path.is_file():
            try:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                for key in ("model", "replicate", "started_utc"):
                    if key in trace:
                        payload[key] = trace[key]
                # The task block matters more than it looks: every reporting
                # path refuses to pool prompt versions, so a stub without one
                # makes the whole tag unanalysable until someone passes
                # --allow-mixed-prompt and pools versions for real.
                prompt = trace.get("prompt") or {}
                task_path = prompt.get("path")
                task_block = {"path": task_path}
                if task_path:
                    try:
                        from autotokamak.bench.taskspec import TaskSpec

                        spec = TaskSpec.from_yaml(Path(task_path))
                        task_block["task_id"] = spec.task_id
                        task_block["prompt_version"] = spec.prompt_version
                    except Exception:  # noqa: BLE001
                        pass
                    payload["task"] = task_block
                if prompt.get("model") and "model" not in payload:
                    payload["model"] = prompt["model"]
            except Exception:  # noqa: BLE001
                pass
        if args.dry_run:
            print(f"would write stub: {run_dir}/result.json")
        else:
            (run_dir / "result.json").write_text(json.dumps(payload, indent=2),
                                                 encoding="utf-8")
        written += 1
    print(f"[reconcile] {written} killed cell(s) "
          f"{'would be' if args.dry_run else ''} recorded")
    return 0


def cmd_spend(args) -> int:
    total, unpriced = measured_spend(args.tag, harness=args.harness)
    print(f"{total:.2f}")
    if args.verbose and unpriced:
        print(f"  unpriced runs (cost not yet recoverable): {len(unpriced)}",
              file=sys.stderr)
    if args.budget is not None and total >= args.budget:
        print(f"[guard] spend ${total:.2f} has reached the ${args.budget:.2f} "
              f"ceiling", file=sys.stderr)
        return 2
    return 0


def cmd_preflight(args) -> int:
    """Everything that is cheaper to discover now than at 2am."""
    problems, notes = [], []
    harnesses = [h.split(":")[0] for h in args.harnesses.split()]

    for h in harnesses:
        if h in NON_OPENAI and not args.allow_non_openai:
            problems.append(
                f"harness '{h}' cannot run an OpenAI model pin — it would spend "
                f"on another provider and confound the harness axis with the "
                f"model axis. Drop it, or pass --allow-non-openai deliberately.")
        env, cli = HARNESS_REQUIREMENTS.get(h, (None, None))
        if env and not os.environ.get(env):
            problems.append(f"harness '{h}' needs ${env}, which is not set "
                            f"(is .env loaded?)")
        if cli and not shutil.which(cli):
            problems.append(f"harness '{h}' needs '{cli}' on PATH — not found")
        if h not in HARNESS_REQUIREMENTS:
            problems.append(f"unknown harness '{h}'")

    if args.task:
        task = Path(args.task)
        if not task.is_file():
            problems.append(f"task file missing: {task}")
        else:
            try:
                from autotokamak.bench.taskspec import TaskSpec

                spec = TaskSpec.from_yaml(task)
                pinned = set(spec.model or {})
                missing_pin = [h for h in harnesses
                               if h not in pinned and h not in {"echo"}]
                if missing_pin:
                    problems.append(
                        f"task {task.name} pins models for {sorted(pinned)} but "
                        f"not for {missing_pin} — those cells would run the "
                        f"adapter's default model, which is not comparable")
                notes.append(f"task {spec.task_id} prompt_version={spec.prompt_version} "
                             f"timeout={spec.timeout_seconds}s "
                             f"feedback_rounds={spec.feedback_rounds}")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"task file will not load: {exc}")

    for asset in ("test_set.h5", "eval_grid.json", "test_params.json"):
        p = REPO_ROOT / "benchmarks" / "assets" / asset
        if not p.is_file():
            problems.append(f"frozen asset missing: benchmarks/assets/{asset} "
                            f"— runs would be unscoreable")

    free_gb = shutil.disk_usage(REPO_ROOT).free / 2**30
    if free_gb < args.min_disk_gb:
        problems.append(
            f"only {free_gb:.1f} GB free; a campaign writes ~20-110 MB of "
            f"datasets per run and a full disk fails solves mid-flight "
            f"(need >= {args.min_disk_gb} GB)")
    else:
        notes.append(f"disk free: {free_gb:.1f} GB")

    # A dirty tree means the archived runs cannot be tied to a commit.
    try:
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=30).stdout
        src_dirty = [ln for ln in dirty.splitlines()
                     if ln[3:].startswith(("src/", "benchmarks/tasks/",
                                           "tools/run_campaign.sh"))]
        if src_dirty:
            notes.append(f"WARNING: {len(src_dirty)} uncommitted change(s) under "
                         f"src/ or benchmarks/tasks/ — commit first so the "
                         f"campaign is tied to a revision")
        else:
            head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
            notes.append(f"HEAD {head}, no uncommitted src/task changes")
    except Exception:  # noqa: BLE001
        pass

    for n in notes:
        print(f"[preflight] {n}")
    for p in problems:
        print(f"[preflight] BLOCKER: {p}", file=sys.stderr)
    if problems:
        print(f"[preflight] {len(problems)} blocker(s) — not spending anything.",
              file=sys.stderr)
        return 1
    print("[preflight] all checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight", help="Check everything before spending")
    p.add_argument("--harnesses", required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--min-disk-gb", type=float, default=10.0)
    p.add_argument("--allow-non-openai", action="store_true")
    p.set_defaults(fn=cmd_preflight)

    p = sub.add_parser("completed", help="Exit 0 if this cell already finished")
    p.add_argument("--tag", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--rep", type=int, default=None)
    p.set_defaults(fn=cmd_completed)

    p = sub.add_parser("reconcile",
                       help="Write a stub result.json for cells killed mid-run")
    p.add_argument("--tag", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_reconcile)

    p = sub.add_parser("spend", help="Dollars spent so far under a tag")
    p.add_argument("--tag", required=True)
    p.add_argument("--budget", type=float, default=None,
                   help="Exit 2 when spend has reached this ceiling")
    p.add_argument("--harness", default=None,
                   help="Count only this substrate's runs (across levels)")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_spend)

    p = sub.add_parser("ratelimits",
                       help="This key's real limits vs measured session rates")
    p.add_argument("--model", default="gpt-5.2")
    p.add_argument("--tag", default=None,
                   help="Tag whose cost_report.csv supplies measured rates")
    p.add_argument("--parallel", type=int, default=None)
    p.add_argument("--headroom", type=float, default=0.6,
                   help="Fraction of the limit to plan against (default 0.6)")
    p.add_argument("--tpm", type=float, default=None, help="Skip the probe")
    p.add_argument("--rpm", type=float, default=None)
    p.add_argument("--tpd", type=float, default=None,
                   help="Override the committed daily cap")
    p.set_defaults(fn=cmd_ratelimits)

    p = sub.add_parser("tokens",
                       help="Tokens charged to this key under a tag, vs the TPD cap")
    p.add_argument("--tag", required=True)
    p.add_argument("--model", default="gpt-5.2")
    p.add_argument("--tpd", type=float, default=None)
    p.add_argument("--since-hours", type=float, default=None,
                   help="Only runs touched in this window (TPD resets daily)")
    p.add_argument("--headroom", type=float, default=0.85,
                   help="Exit 2 at this fraction of the daily cap")
    p.set_defaults(fn=cmd_tokens)

    p = sub.add_parser("forecast", help="Worst-case spend and wall clock")
    p.add_argument("--harnesses", required=True)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--levels", default="L3 L2")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--harness-timeout", default="",
                   help='e.g. "ursa=2700" — truncates that row\'s worst case')
    p.add_argument("--harness-budget", default="",
                   help='e.g. "ursa=45" — caps that row\'s worst case')
    p.add_argument("--model", default="gpt-5.2",
                   help="Which model's daily cap to check the plan against")
    p.add_argument("--tpd", type=float, default=None)
    p.set_defaults(fn=cmd_forecast)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.exit(main())
