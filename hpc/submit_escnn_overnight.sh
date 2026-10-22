#!/usr/bin/env bash
# Run ONCE from the login node. Queues CPU environment setup followed by
# serialized checkpoint-resuming GPU chunks (no login session required).
set -euo pipefail

: "${PIPELINE_CONFIG:?Export PIPELINE_CONFIG=/absolute/path/to/site.env}"
[[ "$PIPELINE_CONFIG" == /* && "$PIPELINE_CONFIG" != *,* ]] || {
  echo 'PIPELINE_CONFIG must be an absolute path without commas' >&2; exit 2;
}
source "$PIPELINE_CONFIG"
cd "${REPO_DIR:?Set REPO_DIR in site.env}"
: "${ESCNN_OUTPUT:?Set ESCNN_OUTPUT in site.env}"
: "${WORK_ROOT:?Set WORK_ROOT in site.env}"
: "${ESCNN_STAGE2G_CACHE:?Set ESCNN_STAGE2G_CACHE in site.env}"
[[ -f "$ESCNN_STAGE2G_CACHE/cache.json" ]] || { echo 'Stage-2G cache not found' >&2; exit 2; }
[[ -f hpc/pbs/setup_escnn_cpu.pbs && -f hpc/pbs/escnn_d8_gpu_chunk.pbs ]] || {
  echo 'Extract helper scripts into REPO_DIR before submitting' >&2; exit 2;
}

# Site-specific account flag is optional. Set ESCNN_PBS_PROJECT in site.env
# only if PBS accepts qsub -P and an account is required.
qsub_extra=()
if [[ -n "${ESCNN_PBS_PROJECT:-}" ]]; then
  qsub_extra=(-P "$ESCNN_PBS_PROJECT")
fi

jobs="${ESCNN_GPU_JOBS:-6}"
[[ "$jobs" =~ ^[1-9][0-9]*$ && "$jobs" -le 12 ]] || {
  echo 'ESCNN_GPU_JOBS must be an integer between 1 and 12' >&2; exit 2;
}

logs="$WORK_ROOT/logs/escnn-overnight"
mkdir -p "$logs"
# Reserve the log directory; do not submit an accidental second chain.
if ! mkdir "$logs/.submitted" 2>/dev/null; then
  echo "Submission marker already exists: $logs/.submitted" >&2
  echo 'Check qstat before deciding whether to submit another chain.' >&2
  exit 2
fi

# Cleanup marker only if nothing at all has been submitted.
submitted=0
trap 'if [[ "$submitted" == 0 ]]; then rmdir "$logs/.submitted" 2>/dev/null || true; fi' EXIT

cpu_id=$(qsub "${qsub_extra[@]}" -o "$logs/cpu-setup.log" \
  -v "PIPELINE_CONFIG=$PIPELINE_CONFIG" hpc/pbs/setup_escnn_cpu.pbs)
submitted=1
printf 'CPU installation job: %s\n' "$cpu_id"
printf '%s\n' "$cpu_id" > "$logs/cpu.jobid"

prev="$cpu_id"
for ((i=1;i<=jobs;i++)); do
  gpu_id=$(qsub "${qsub_extra[@]}" -o "$logs/gpu-$i.log" \
    -W "depend=afterok:$prev" \
    -v "PIPELINE_CONFIG=$PIPELINE_CONFIG" hpc/pbs/escnn_d8_gpu_chunk.pbs)
  printf 'GPU chunk %d: %s  (afterok %s)\n' "$i" "$gpu_id" "$prev"
  printf '%s\n' "$gpu_id" > "$logs/gpu-$i.jobid"
  prev="$gpu_id"
done
printf '\nLogs: %s\n' "$logs"
printf 'Track jobs: qstat -u "$USER"\n'
printf 'First see CPU setup results: %s/cpu-setup.log\n' "$logs"
