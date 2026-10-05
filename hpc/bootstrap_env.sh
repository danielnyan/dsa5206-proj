#!/usr/bin/env bash
set -euo pipefail
source "${PIPELINE_CONFIG:?Set PIPELINE_CONFIG to site.env}"
if [[ -n "${SITE_SETUP:-}" ]]; then eval "$SITE_SETUP"; fi
python3 -m venv "${VENV_DIR:?}"
"$VENV_DIR/bin/python" -m pip install -r "$REPO_DIR/requirements-reimplementation-wsl.txt" -r "$REPO_DIR/hpc/requirements-extra.txt"
echo 'Environment installed. Run model feasibility checks in a GPU PBS allocation.'
