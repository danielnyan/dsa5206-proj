#!/usr/bin/env bash
# Invoke from a login node. Downloads must have completed first.
set -euo pipefail
: "${PIPELINE_CONFIG:?Set absolute site.env path}"
source "$PIPELINE_CONFIG"
mode=${MODE:-smoke}
case "$mode" in smoke|production) ;; *) exit 2;; esac
base="$REPO_DIR/hpc/pbs"
dependency=()
if [[ "${SKIP_PREP:-0}" != 1 ]]; then
  prep_id=$(qsub -v "PIPELINE_CONFIG=$PIPELINE_CONFIG" "$base/prepare_gleason.pbs")
  dependency=(-W "depend=afterok:$prep_id")
  echo "Preparation job: $prep_id"
fi
train_id=$(qsub "${dependency[@]}" -v "PIPELINE_CONFIG=$PIPELINE_CONFIG,MODE=$mode" "$base/train_gleason.pbs")
echo "Training job: $train_id"
echo "Inspect $RUN_ROOT/${train_id}-${mode} before scheduling evaluation."
