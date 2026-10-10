#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/m6_env.sh"
CONTROL="$PROJECT/runs/m6_pipeline_e${EPOCHS:-5}"
mkdir -p "$CONTROL"
rm -f "$CONTROL/pipeline.exit_code"
on_exit() { code=$?; printf '%s\n' "$code" > "$CONTROL/pipeline.exit_code"; }
trap on_exit EXIT
printf 'preparing_cache\n' > "$CONTROL/phase.txt"
if [[ "$M6_DIRECTION_MODE" == unfiltered ]]; then
  py scripts/promote_m6_unfiltered.py
else
  bash scripts/prepare_m6.sh
fi
printf 'training\n' > "$CONTROL/phase.txt"
bash scripts/train_m6.sh "${EPOCHS:-5}"
printf 'complete\n' > "$CONTROL/phase.txt"
