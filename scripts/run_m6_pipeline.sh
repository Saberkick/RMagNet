#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/m6_env.sh"
CONTROL="$PROJECT/runs/m6_pipeline_e${EPOCHS:-5}"
mkdir -p "$CONTROL"
on_exit() { code=$?; printf '%s\n' "$code" > "$CONTROL/pipeline.exit_code"; }
trap on_exit EXIT
printf 'preparing_cache\n' > "$CONTROL/phase.txt"
bash scripts/prepare_m6.sh
printf 'training\n' > "$CONTROL/phase.txt"
bash scripts/train_m6.sh "${EPOCHS:-5}"
printf 'complete\n' > "$CONTROL/phase.txt"
