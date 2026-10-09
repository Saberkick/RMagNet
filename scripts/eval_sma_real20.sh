#!/usr/bin/env bash
set -euo pipefail
source /share/linmingheng-local/xuke/RMagNet/scripts/sma_env.sh
export CUDA_VISIBLE_DEVICES=0,1,2
check_gpus
RUN="$PROJECT/runs/real20_sma_dataset2_e50"
[[ ! -e "$RUN" ]] || { echo "Output already exists: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
pids=()
for spec in '0 best' '1 latest' '2 m4best'; do
  read -r gpu variant <<< "$spec"
  args=()
  if [[ "$variant" != m4best ]]; then
    args=(--checkpoint "$PROJECT/runs/sma_dataset2_e50/${variant}_sma.safetensors")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" uvpython -m src.rmagnet.sma_real20 "${args[@]}" --output "$RUN/$variant" > "$RUN/logs/$variant.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status == 0 )); then
  uvpython -m src.rmagnet.sma_real20_report > "$RUN/report.console.log"
fi
exit "$status"
