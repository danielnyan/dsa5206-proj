#!/bin/bash
# Submit N chained train_model2.pbs jobs (each starts after the previous ends, whatever its exit),
# instead of resubmitting from a compute node. Links after completion exit immediately.
#   bash ens_model_training/jobs/submit_train_chain.sh N [BATCH=4] [extra qsub -v vars, e.g. RUN_NAME=x]
# Example (option b, the chosen path):  bash ens_model_training/jobs/submit_train_chain.sh 3
set -euo pipefail
source "$(dirname "$0")/config.sh"
count="${1:?number of jobs}"; batch="${2:-4}"; extra="${3:-}"
[[ "$NSCC_PROJECT" == CHANGE_ME ]] && { echo "Set NSCC_PROJECT in ens_model_training/jobs/config.sh first" >&2; exit 2; }
vars="CODE_DIR=$CODE,BATCH=$batch${extra:+,$extra}"
previous=""
for ((i = 1; i <= count; i++)); do
    depend=(); [[ -n "$previous" ]] && depend=(-W "depend=afterany:$previous")
    previous=$(qsub -P "$NSCC_PROJECT" -q "$GPU_QUEUE" ${depend[@]+"${depend[@]}"} -v "$vars" "$CODE/nscc/train_model2.pbs")
    echo "link $i: $previous"
done
