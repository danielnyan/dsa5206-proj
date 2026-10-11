#!/usr/bin/env bash
# Lightweight login-node submission only; downloads/processing stay in PBS.
set -euo pipefail
: "${PIPELINE_CONFIG:?Set absolute site.env path}"
requested_dataset=${1:-${DATASET:-crowd-sicap}}
action=${2:-${MODE:-smoke}}
case "$requested_dataset" in crowd-sicap|gleason2019|panda) ;; *) echo 'Dataset must be crowd-sicap, gleason2019 or panda' >&2; exit 2;; esac
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

if [[ "$DATASET" == panda ]]; then
  : "${PANDA_CHUNKS_DIR:?Set PANDA_CHUNKS_DIR to the directory containing panda_150um_chunk_XXXX.zip files}"
  [[ -d "$PANDA_CHUNKS_DIR" ]] || { echo "PANDA_CHUNKS_DIR does not exist: $PANDA_CHUNKS_DIR" >&2; exit 2; }
  compgen -G "$PANDA_CHUNKS_DIR/panda_150um_chunk_*.zip" >/dev/null || {
    echo "No panda_150um_chunk_XXXX.zip files found in $PANDA_CHUNKS_DIR" >&2; exit 2;
  }
  [[ "$action" != mine ]] || { echo 'PANDA transfer shards contain no mixed-region mining pool; mine is unsupported for DATASET=panda.' >&2; exit 2; }
fi

if [[ -f "$PREPARED_DIR/prepared.json" ]]; then
  "$VENV_DIR/bin/python" - "$PREPARED_DIR/prepared.json" "$DATASET" "${PANDA_ALLOW_PARTIAL:-0}" "${PANDA_CHUNKS_DIR:-}" <<'PY'
import hashlib, json, pathlib, re, sys
prepared_path, dataset, allow_partial, chunks_dir = sys.argv[1:]
p = json.load(open(prepared_path, encoding='utf-8'))
allow_partial = allow_partial == '1'
ok = bool(p.get('complete'))
if dataset in ('gleason2019', 'panda'):
    ok = ok and bool(p.get('training_ready'))
if p.get('sample_only'):
    ok = ok and dataset == 'panda' and allow_partial
if not ok:
    raise SystemExit('Existing preparation is incomplete/audit-only, lacks seed classes, or is partial without PANDA_ALLOW_PARTIAL=1; choose a new PREPARED_DIR or inspect the audit.')
if dataset == 'panda':
    if p.get('datasets') != ['panda']:
        raise SystemExit('Existing PREPARED_DIR is not a PANDA preparation.')
    # Immutable-preparation guard: the uploaded ZIP set must match the one that
    # produced this PREPARED_DIR. If more chunks are uploaded later, use a new
    # PREPARED_DIR so preliminary cohorts remain reproducible.
    root = pathlib.Path(chunks_dir)
    rx = re.compile(r'^panda_150um_chunk_(\d{4})\.zip$')
    current = []
    for path in sorted(root.iterdir()):
        m = rx.fullmatch(path.name)
        if not (path.is_file() and m):
            continue
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                h.update(block)
        current.append((int(m.group(1)), path.name, h.hexdigest()))
    frozen = sorted((int(x['chunk_index']), x['filename'], x['sha256']) for x in p.get('chunk_archives', []))
    if current != frozen:
        raise SystemExit(
            'PANDA_CHUNKS_DIR differs from the immutable chunk set recorded in PREPARED_DIR. '
            'Do not silently extend an old cohort: set PREPARED_DIR to a fresh path and resubmit.'
        )
PY
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
