#!/usr/bin/env bash
set -euo pipefail
PROJECT=/share/linmingheng-local/xuke/RMagNet
cd "$PROJECT"
session=sma_dataset2_e50
tmux has-session -t "$session" 2>/dev/null && { echo 'Session already exists'; exit 2; }
mkdir -p runs/sma_launch
[[ ! -e runs/sma_launch/dataset2_e50.exit_code ]] || { echo 'Existing completed launch; refusing accidental rerun'; exit 3; }
tmux new-session -d -s "$session" -c "$PROJECT" \
  'bash scripts/run_sma_dataset2.sh > runs/sma_launch/dataset2_e50.console.log 2>&1; code=$?; printf "%s\n" "$code" > runs/sma_launch/dataset2_e50.exit_code'
tmux display-message -p -t "$session" '#{pane_pid}' > runs/sma_launch/dataset2_e50.pid
printf 'Started tmux %s; log: %s/runs/sma_launch/dataset2_e50.console.log\n' "$session" "$PROJECT"
