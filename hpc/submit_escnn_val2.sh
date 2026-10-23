#!/usr/bin/env bash
# Run ONCE from login node; CPU dependency setup then fixed-checkpoint GPU VAL2.
set -euo pipefail
: "${PIPELINE_CONFIG:?export PIPELINE_CONFIG=/absolute/path/to/site.env}"
[[ "$PIPELINE_CONFIG" == /* && "$PIPELINE_CONFIG" != *,* ]] || {
  echo 'PIPELINE_CONFIG must be an absolute path without commas' >&2; exit 2;
}
source "$PIPELINE_CONFIG"
cd "${REPO_DIR:?Set REPO_DIR in site.env}"
: "${ESCNN_OUTPUT:?Set ESCNN_OUTPUT in site.env}"
: "${ESCNN_VAL2_EXTRACTED:?Set ESCNN_VAL2_EXTRACTED in site.env}"
: "${WORK_ROOT:?Set WORK_ROOT in site.env}"
[[ -f "$ESCNN_OUTPUT/best.pt" ]] || { echo 'Missing selected best.pt' >&2; exit 2; }
[[ -f hpc/pbs/setup_escnn_val2_cpu.pbs && -f hpc/pbs/evaluate_escnn_d8_val2.pbs ]] || {
  echo 'ZIP was not extracted into repository root' >&2; exit 2;
}
[[ -d "$ESCNN_VAL2_EXTRACTED/val_dataset_2_norm" && -d "$ESCNN_VAL2_EXTRACTED/val_dataset_2_tu" ]] || {
  echo 'VAL2 extracted roots missing' >&2; exit 2;
}
# Unique attempts are deliberate. Never delete a partial previous evaluation.
OUT=${ESCNN_VAL2_OUTPUT:-"$ESCNN_OUTPUT/val2-stage2g-$(date -u +%Y%m%dT%H%M%SZ)"}
[[ "$OUT" == /* && "$OUT" != *,* ]] || { echo 'Output must be absolute and contain no comma' >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Output already exists: $OUT" >&2; exit 2; }
LOGDIR="$WORK_ROOT/logs/escnn-val2"
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
[[ "$OUT" != "$ESCNN_VAL2_EXTRACTED" && "$OUT" != "$ESCNN_OUTPUT" ]] || {
  echo 'Unsafe output directory' >&2; exit 2;
}
extra=()
if [[ -n "${ESCNN_PBS_PROJECT:-}" ]]; then extra+=(-P "$ESCNN_PBS_PROJECT"); fi
cpu_id=$(qsub "${extra[@]}" -o "$LOGDIR" -e "$LOGDIR" -v "PIPELINE_CONFIG=$PIPELINE_CONFIG" \
  hpc/pbs/setup_escnn_val2_cpu.pbs)
[[ -n "$cpu_id" ]] || { echo 'CPU submission failed' >&2; exit 1; }
echo "CPU dependency setup: $cpu_id"
gpu_id=$(qsub "${extra[@]}" -W "depend=afterok:$cpu_id" -o "$LOGDIR" -e "$LOGDIR" \
  -v "PIPELINE_CONFIG=$PIPELINE_CONFIG,ESCNN_VAL2_OUTPUT=$OUT" hpc/pbs/evaluate_escnn_d8_val2.pbs)
echo "Dependent GPU evaluation: $gpu_id"
echo "Selected checkpoint: $ESCNN_OUTPUT/best.pt"
echo "Expected VAL2 results: $OUT/metrics.json"
echo "PBS manages these jobs after logout. If setup fails, afterok prevents GPU use."
