#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/sma_env.sh"
EPOCHS="${1:-${EPOCHS:-20}}"
SESSION="${SESSION:-sma_e${EPOCHS}}"
[[ "$SESSION" =~ ^[A-Za-z0-9_-]+$ && "$EPOCHS" =~ ^[1-9][0-9]*$ && "$EPOCHS" -le 20 ]] || exit 2
command -v tmux >/dev/null
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session exists: $SESSION" >&2; exit 3; fi
RUN_DIR="${RUN_DIR:-$PROJECT/runs/sma_e${EPOCHS}}"
[[ ! -e "$RUN_DIR" ]] || { echo "Run already exists: $RUN_DIR" >&2; exit 4; }
export RUN_DIR EPOCHS
mkdir -p "$PROJECT/runs/sma_launch"
printf -v train_command '%q ' env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "EPOCHS=$EPOCHS" "RUN_DIR=$RUN_DIR" "OMP_NUM_THREADS=$OMP_NUM_THREADS" "NUM_WORKERS=${NUM_WORKERS:-1}" "MAX_STEPS=${MAX_STEPS:-0}" "EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-0}" "LEARNING_RATE=${LEARNING_RATE:-1e-4}" "CACHE_ROOT=$CACHE_ROOT" "MEMORY_DIR=$MEMORY_DIR" bash scripts/train_sma.sh
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && $train_command > 'runs/sma_launch/${SESSION}.console.log' 2>&1; status=\$?; echo \$status > 'runs/sma_launch/${SESSION}.exitcode'"
tmux display-message -pt "$SESSION" '#{pane_pid}' > "$PROJECT/runs/sma_launch/${SESSION}.pid"
echo "Started $SESSION; log: $PROJECT/runs/sma_launch/${SESSION}.console.log"
