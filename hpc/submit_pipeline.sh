#!/usr/bin/env bash
# Lightweight login-node submission only; downloads/processing stay in PBS.
set -euo pipefail
: "${PIPELINE_CONFIG:?Set absolute site.env path}"
requested_dataset=${1:-${DATASET:-crowd-sicap}}
action=${2:-${MODE:-smoke}}
case "$requested_dataset" in crowd-sicap|gleason2019) ;; *) echo 'Dataset must be crowd-sicap or gleason2019' >&2; exit 2;; esac
export DATASET="$requested_dataset"
source "$PIPELINE_CONFIG"
if [[ -n "${SITE_SETUP:-}" ]]; then eval "$SITE_SETUP"; fi
if [[ "$DATASET" != "$requested_dataset" ]]; then
  echo 'Old configuration overrides DATASET. Refresh from hpc/site.env.example.' >&2; exit 2
fi
case "$action" in prepare|smoke|train|production|evaluate|mine) ;; *) echo 'Action: prepare, smoke, train, evaluate, mine' >&2; exit 2;; esac
[[ "$PIPELINE_CONFIG" == /* && "$PIPELINE_CONFIG" != *,* ]] || { echo 'Use an absolute config path without commas' >&2; exit 2; }
[[ -x "$VENV_DIR/bin/python" ]] || { echo 'Run hpc/bootstrap_env.sh once first' >&2; exit 2; }
mkdir -p "$RUN_ROOT"
if [[ -f "$PREPARED_DIR/prepared.json" ]]; then
  "$VENV_DIR/bin/python" -c 'import json,sys; p=json.load(open(sys.argv[1])); ok=p.get("complete") and not p.get("sample_only") and (sys.argv[2]!="gleason2019" or p.get("training_ready")); sys.exit(0 if ok else "Existing preparation is incomplete/audit-only or lacks seed classes; choose a new PREPARED_DIR or inspect the audit.")' "$PREPARED_DIR/prepared.json" "$DATASET"
fi
base="$REPO_DIR/hpc/pbs"
vars="PIPELINE_CONFIG=$PIPELINE_CONFIG,DATASET=$DATASET"
case "$action" in
  evaluate|mine)
    : "${MODEL_DIR:?Set MODEL_DIR to a completed production model from the selected track}"
    mode=production
    [[ "$action" == mine ]] && mode=mine
    qsub -v "$vars,MODE=$mode,MODEL_DIR=$MODEL_DIR" "$base/evaluate_gleason.pbs"
    exit 0;;
  train|production)
    [[ -f "$PREPARED_DIR/prepared.json" ]] || { echo 'Run preparation/smoke first' >&2; exit 2; }
    qsub -v "$vars,MODE=production" "$base/train_gleason.pbs"
    exit 0;;
esac
dependency=()
if [[ -f "$PREPARED_DIR/prepared.json" ]]; then
  echo "Reusing preparation: $PREPARED_DIR"
elif [[ -e "$PREPARED_DIR" ]]; then
  echo "Incomplete preparation: choose a new PREPARED_DIR; existing files are preserved." >&2; exit 2
else
  if [[ "$DATASET" == crowd-sicap && "${TRAINING_COHORT:-pure}" == pure ]]; then
    [[ -f "${SICAP_WSI_LABELS:-}" ]] || { echo 'Stage SICAP wsi_labels.xlsx and set SICAP_WSI_LABELS first' >&2; exit 2; }
  fi
  prep_id=$(qsub -v "$vars" "$base/prepare_gleason.pbs")
  dependency=(-W "depend=afterok:$prep_id")
  echo "Preparation job: $prep_id"
fi
[[ "$action" == prepare ]] && exit 0
train_id=$(qsub "${dependency[@]}" -v "$vars,MODE=smoke" "$base/train_gleason.pbs")
echo "GPU smoke job: $train_id"
echo "Inspect $RUN_ROOT/${train_id}-smoke before submitting train."
