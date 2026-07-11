#!/usr/bin/env bash
# Collect meta-agent traces for GEPA optimization.
#
# Runs the meta-loop N times with varied configs (seeds + max-iterations) so
# the resulting trainset covers a diversity of states/decisions, not just one
# pattern. Output traces land in experiments/<run_id>/trace.json with linked
# meta_trace.json under each workspace.
#
# Usage:
#   tools/collect_traces.sh                                 # N=15, gpt-5-mini
#   tools/collect_traces.sh --n 30 --model openai:gpt-5.2   # bigger / pricier
#   tools/collect_traces.sh --use-baseline --tag baseline   # A/B baseline arm
#
# Cost estimate (gpt-5-mini, default settings):
#   N=15  -> ~$5-15  (depending on how often the agent picks extend_search)
#   N=30  -> ~$10-30

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

N=15
MODEL="openai:gpt-5-mini"
MAX_ITERS_MIN=1
MAX_ITERS_MAX=3
USE_BASELINE=""
TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)
            N="$2"; shift 2 ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --max-iters-min)
            MAX_ITERS_MIN="$2"; shift 2 ;;
        --max-iters-max)
            MAX_ITERS_MAX="$2"; shift 2 ;;
        --use-baseline)
            USE_BASELINE="--use-baseline"; shift ;;
        --tag)
            TAG="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,16p' "$0"
            exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 2 ;;
    esac
done

if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    set -a; source .env; set +a
fi
# shellcheck disable=SC1091
source venv/bin/activate

EXPERIMENTS_DIR="experiments"
mkdir -p "$EXPERIMENTS_DIR"

# Per-run experiments_dir subdirectory if tagged, so baseline/optimized
# A/B runs don't intermix with the optimization trainset.
if [[ -n "$TAG" ]]; then
    EXPERIMENTS_DIR="experiments_${TAG}"
    mkdir -p "$EXPERIMENTS_DIR"
fi

echo "Collecting $N traces using model=$MODEL"
echo "  max_iterations rotates in [$MAX_ITERS_MIN, $MAX_ITERS_MAX]"
echo "  experiments dir: $EXPERIMENTS_DIR"
[[ -n "$USE_BASELINE" ]] && echo "  forcing baseline picker (no optimized JSON)"
echo ""

start=$(date +%s)
for ((i=0; i<N; i++)); do
    # Rotate max_iterations across the requested range so the trainset has
    # both short (terminate-early) and longer (extend/regen) trajectories.
    if (( MAX_ITERS_MIN == MAX_ITERS_MAX )); then
        ITERS=$MAX_ITERS_MIN
    else
        ITERS=$(( MAX_ITERS_MIN + (i % (MAX_ITERS_MAX - MAX_ITERS_MIN + 1)) ))
    fi
    echo "--- run $((i+1))/$N  max-iterations=$ITERS ---"
    PYTHONPATH=src/autotokamak python -m agent.runners.meta_loop \
        --config src/autotokamak/agent/prompts/surrogate_meta.yaml \
        --model "$MODEL" \
        --max-iterations "$ITERS" \
        --experiments-dir "$EXPERIMENTS_DIR" \
        $USE_BASELINE \
        || { echo "Run $((i+1)) FAILED; continuing"; continue; }
done

elapsed=$(( $(date +%s) - start ))
n_traces=$(find "$EXPERIMENTS_DIR" -maxdepth 2 -name trace.json 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "=== Done ==="
echo "  elapsed:   ${elapsed}s"
echo "  traces in $EXPERIMENTS_DIR: $n_traces"
echo ""
echo "Next: run GEPA optimization"
echo "  PYTHONPATH=src/autotokamak python -m autotokamak.agent.dspy.optimize_meta \\"
echo "      --experiments-dir $EXPERIMENTS_DIR \\"
echo "      --output src/autotokamak/agent/dspy/optimized/meta_picker.json \\"
echo "      --auto medium"
