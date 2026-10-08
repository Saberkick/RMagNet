#!/usr/bin/env bash
set -euo pipefail
PROJECT=/share/linmingheng-local/xuke/RMagNet
SESSION=sma_gtnoise5_e20_continue
source "$PROJECT/scripts/sma_env.sh"
check_gpus
if tmux has-session -t "$SESSION" 2>/dev/null; then echo 'Session already exists' >&2; exit 3; fi
[[ ! -e "$PROJECT/runs/sma_gtnoise5_e20_continue" ]] || { echo 'Run exists' >&2; exit 4; }
mkdir -p "$PROJECT/runs/sma_launch"
printf -v command '%q ' env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "OMP_NUM_THREADS=$OMP_NUM_THREADS" "NUM_WORKERS=${NUM_WORKERS:-1}" bash scripts/continue_sma_gtnoise.sh
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && $command > 'runs/sma_launch/${SESSION}.console.log' 2>&1; status=\$?; echo \$status > 'runs/sma_launch/${SESSION}.exitcode'"
tmux display-message -pt "$SESSION" '#{pane_pid}' > "$PROJECT/runs/sma_launch/${SESSION}.pid"
echo "Started $SESSION; log: $PROJECT/runs/sma_launch/${SESSION}.console.log"
