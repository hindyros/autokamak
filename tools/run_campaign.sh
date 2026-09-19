#!/usr/bin/env bash
# Replicated benchmark campaign driver.
#
# Extends the one-shot experiments/matrix-*/run_matrix.sh pattern with the
# thing the paper needs and those lacked: REPLICATES. Each (level, harness)
# cell is run --reps times, and every run records its replicate index, so
# tools/aggregate_matrix.py can group by cell and put error bars on it.
#
# Levels run as separate waves (all L3, then all L2) so a rate-limit stall
# in one wave cannot leave a level half-finished.
#
#   tools/run_campaign.sh --tag armA --reps 5 --parallel 3
#   tools/run_campaign.sh --tag armA --reps 5 --resume        # after a crash
#   tools/run_campaign.sh --tag armB --reps 5 --harnesses "ursa dspy pi"
#
# --dry-run prints every command without spending anything.
#
# Three money guards, all added after they cost something:
#   --resume       skip (condition, rep) pairs that already COMPLETED under
#                  this tag. Without it a crash at hour 18 re-runs and
#                  re-pays for every finished cell, because bench run always
#                  mints a new run_id.
#   --budget-usd   stop launching new cells once recorded spend reaches a
#                  ceiling. Cells in flight are allowed to finish.
#   preflight      keys, CLIs, model pins, frozen assets, disk and git state
#                  are checked BEFORE the first paid call
#                  (tools/campaign_guard.py; --skip-preflight to bypass).
#
# claude_sdk is NOT in the default harness list: this campaign pins an
# OpenAI model across substrates, and that adapter is Anthropic by
# construction, so including it would spend on another provider and confound
# the harness axis with the model axis.
set -uo pipefail
cd "$(dirname "$0")/.."

TAG=""; REPS=5; PARALLEL=3; MODEL=""; DRY=0; TIMEOUT=""
RESUME=0; BUDGET=""; SKIP_PREFLIGHT=0; MIN_DISK_GB=10
HARNESSES="ursa dspy pi cursor"
LEVELS="L3 L2"
TASK_SUFFIX="mini_v3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)        TAG="$2"; shift 2 ;;
    --reps)       REPS="$2"; shift 2 ;;
    --parallel)   PARALLEL="$2"; shift 2 ;;
    --model)      MODEL="$2"; shift 2 ;;
    --timeout)    TIMEOUT="$2"; shift 2 ;;   # per-run cap, overrides the task

    --harnesses)  HARNESSES="$2"; shift 2 ;;
    --levels)     LEVELS="$2"; shift 2 ;;
    --task-suffix) TASK_SUFFIX="$2"; shift 2 ;;
    --resume)     RESUME=1; shift ;;
    --budget-usd) BUDGET="$2"; shift 2 ;;
    --min-disk-gb) MIN_DISK_GB="$2"; shift 2 ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --dry-run)    DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TAG" ]] || { echo "--tag is required" >&2; exit 2; }

[[ -f venv/bin/activate ]] && source venv/bin/activate
[[ -f .env ]] && { set -a; source .env; set +a; }
export PYTHONUNBUFFERED=1

LOGS="experiments/$TAG/logs"
mkdir -p "$LOGS"
ABORT="$LOGS/.abort"
rm -f "$ABORT"

GUARD="python tools/campaign_guard.py"

# ---- preflight: everything cheaper to find now than at 2am ---------------
if [[ "$SKIP_PREFLIGHT" != "1" && "$DRY" != "1" ]]; then
  first_level="${LEVELS%% *}"
  if ! $GUARD preflight --harnesses "$HARNESSES" \
        --task "benchmarks/tasks/${first_level}_${TASK_SUFFIX}.yaml" \
        --min-disk-gb "$MIN_DISK_GB"; then
    echo "Refusing to start. Fix the blockers above, or --skip-preflight if you "         "genuinely mean to." >&2
    exit 3
  fi
fi

# Hard wall-clock ceiling per cell, enforced by the DRIVER.
#
# The in-Python caps are not sufficient on their own. In the first n=10
# attempt URSA absorbed a SIGALRM inside a broad `except Exception` and ran
# 10h against a 90-minute cap, while a pi cell deadlocked in
# subprocess.run(capture_output=True, timeout=...) for 9h19m — the second
# communicate() blocks until grandchildren release the inherited pipes.
# Both are fixed in the adapters, but an agent harness is third-party code
# and the campaign must not be hostage to it. This watchdog kills the cell's
# entire PROCESS GROUP and cannot be swallowed by anything running inside.
#
# Set above the task's own timeout_seconds so the graceful in-Python path
# (which still records cost and partial output) gets first refusal.
CELL_HARD_TIMEOUT="${CELL_HARD_TIMEOUT:-6600}"

run_cell() {
  local name="$1"; shift

  # A budget stop aborts the cells still queued behind it; anything already
  # in flight is left to finish and record its spend.
  if [[ -f "$ABORT" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP  $name ($(cat "$ABORT"))"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START $name"

  # Job control gives this child its own process group (pgid == pid), so
  # `kill -- -$pid` reaps the agent and every solver process it spawned.
  # macOS ships no setsid(1), so this is the portable route.
  set -m
  "$@" >"$LOGS/$name.log" 2>&1 &
  local pid=$!
  set +m

  # Watchdog by POLLING, not by a sleeping subshell.
  #
  # The obvious `( sleep $CAP; kill ... ) &` leaves an orphan behind every
  # cell: bash kills the subshell, its `sleep` survives with PPID 1, and it
  # holds the driver's stdout for the rest of the cap. A piped or tee'd
  # campaign then LOOKS hung for up to two hours after its last cell
  # finished, with one stray process per cell (observed: 80-cell campaign,
  # 80 sleeps). Polling costs a wakeup every 5s and leaves nothing behind.
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if (( waited >= CELL_HARD_TIMEOUT )); then
      echo "[watchdog] hard-killing $name: exceeded ${CELL_HARD_TIMEOUT}s" \
        | tee -a "$LOGS/$name.log"
      kill -TERM -- "-$pid" 2>/dev/null
      sleep 15
      kill -KILL -- "-$pid" 2>/dev/null
      break
    fi
    sleep 5
    waited=$((waited + 5))
  done

  wait "$pid"
  local rc=$?

  echo "$rc" >"$LOGS/$name.exit"
  echo "[$(date +%H:%M:%S)] DONE  $name (exit $rc)"

  # Spend is checked AFTER each cell rather than on a timer: a cell's cost
  # only becomes visible when it writes result.json.
  if [[ -n "${BUDGET:-}" ]]; then
    local spent
    spent=$(python tools/campaign_guard.py spend --tag "$TAG" 2>/dev/null | tail -1)
    echo "[$(date +%H:%M:%S)] spend so far: \$${spent} / \$${BUDGET}"
    if python -c "import sys; sys.exit(0 if float('${spent:-0}') >= float('$BUDGET') else 1)"; then
      echo "budget ceiling \$$BUDGET reached (spent \$$spent)" >"$ABORT"
      echo "[$(date +%H:%M:%S)] BUDGET STOP — no further cells will start"
    fi
  fi
}
export -f run_cell
export LOGS CELL_HARD_TIMEOUT ABORT BUDGET TAG

for level in $LEVELS; do
  task="benchmarks/tasks/${level}_${TASK_SUFFIX}.yaml"
  [[ -f "$task" ]] || { echo "missing task: $task" >&2; exit 2; }
  echo "=== WAVE $level  (task=$task, reps=$REPS, -P $PARALLEL) ==="

  jobs=""
  for rep in $(seq 1 "$REPS"); do
    for h in $HARNESSES; do
      # "name:reps" caps that harness's replicate count (e.g. "ursa:4"),
      # so a pathologically slow substrate does not gate the whole campaign.
      hname="${h%%:*}"; hreps="${h##*:}"
      [[ "$hname" == "$hreps" ]] && hreps="$REPS"
      if (( rep > hreps )); then continue; fi
      h="$hname"
      # --resume: a completed (condition, rep) is not paid for twice.
      if [[ "$RESUME" == "1" ]] && \
         $GUARD completed --tag "$TAG" --condition "${level}-${h}" --rep "$rep" >/dev/null 2>&1; then
        echo "  resume: skipping ${level}-${h} rep ${rep} (already completed)"
        continue
      fi
      cmd="python -m autotokamak.bench run --task $task --harness $h --tag $TAG --rep $rep"
      [[ -n "$MODEL" ]]   && cmd="$cmd --model $MODEL"
      [[ -n "$TIMEOUT" ]] && cmd="$cmd --timeout $TIMEOUT"
      jobs+="${level}-${h}-r${rep} ${cmd}"$'\n'
    done
  done

  if [[ "$DRY" == "1" ]]; then
    printf '%s' "$jobs" | sed 's/^/  would run: /'
    continue
  fi
  printf '%s' "$jobs" | xargs -P "$PARALLEL" -I {} bash -c 'run_cell {}'
  echo "[$(date +%H:%M:%S)] WAVE $level complete"
done

[[ "$DRY" == "1" ]] && exit 0

if [[ -f "$ABORT" ]]; then
  echo "=== CAMPAIGN STOPPED EARLY: $(cat "$ABORT") ==="
  echo "Re-run the same command with --resume (and a higher --budget-usd) to "     "finish the remaining cells without re-paying for the completed ones."
fi

# Cells the watchdog killed never wrote a result.json; without this they
# vanish from every report rather than being counted as timeouts.
$GUARD reconcile --tag "$TAG"

echo "=== EXIT CODES ==="
for f in "$LOGS"/*.exit; do
  printf '%-28s exit %s\n' "$(basename "${f%.exit}")" "$(cat "$f")"
done
echo
total=$($GUARD spend --tag "$TAG" 2>/dev/null | tail -1)
echo "Recorded spend for this tag: \$${total}"
echo
echo "Next: python tools/cost_report.py    --tag $TAG   # prices the token-only cells"
echo "      python tools/aggregate_matrix.py --tag $TAG"
