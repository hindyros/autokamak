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
#   tools/run_campaign.sh --tag armB --reps 5 --model claude-sonnet-5 \
#                         --harnesses "ursa dspy claude_sdk pi"
#
# --dry-run prints every command without spending anything.
set -uo pipefail
cd "$(dirname "$0")/.."

TAG=""; REPS=5; PARALLEL=3; MODEL=""; DRY=0; TIMEOUT=""
HARNESSES="ursa dspy claude_sdk pi cursor"
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
  echo "[$(date +%H:%M:%S)] START $name"

  # Job control gives this child its own process group (pgid == pid), so
  # `kill -- -$pid` reaps the agent and every solver process it spawned.
  # macOS ships no setsid(1), so this is the portable route.
  set -m
  "$@" >"$LOGS/$name.log" 2>&1 &
  local pid=$!
  set +m

  (
    sleep "$CELL_HARD_TIMEOUT"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[watchdog] hard-killing $name: exceeded ${CELL_HARD_TIMEOUT}s"         | tee -a "$LOGS/$name.log"
      kill -TERM -- "-$pid" 2>/dev/null
      sleep 15
      kill -KILL -- "-$pid" 2>/dev/null
    fi
  ) &
  local watchdog=$!

  wait "$pid"
  local rc=$?
  kill "$watchdog" 2>/dev/null
  wait "$watchdog" 2>/dev/null

  echo "$rc" >"$LOGS/$name.exit"
  echo "[$(date +%H:%M:%S)] DONE  $name (exit $rc)"
}
export -f run_cell
export LOGS CELL_HARD_TIMEOUT

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

echo "=== EXIT CODES ==="
for f in "$LOGS"/*.exit; do
  printf '%-28s exit %s\n' "$(basename "${f%.exit}")" "$(cat "$f")"
done
echo
echo "Next: python tools/aggregate_matrix.py --tag $TAG"
