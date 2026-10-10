#!/bin/bash
# One-time environment setup. Run on the LOGIN node (needs internet; compute nodes may not have it):
#   bash ens_model_training/jobs/setup_env.sh
# Creates a conda prefix env at $ENV_DIR with TensorFlow's pip CUDA wheels (no CUDA module needed,
# only a recent NVIDIA driver on the GPU nodes), the pinned preprocessing stack of
# ens_model_training/requirements.txt, and pre-downloads the NASNetLarge ImageNet weights into $KERAS_HOME.
# Unpinned packages resolve to pip's newest compatible release; the result is written to
# $ENV_DIR/nscc_env_freeze.txt. GPU visibility is checked by gpu_probe.pbs, not here.
set -euo pipefail
source "$(dirname "$0")/config.sh"

pin() { [[ -n "$2" ]] && echo "$1==$2" || echo "$1"; }

module load miniforge3 2>/dev/null || module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    conda create -y -p "$ENV_DIR" "python=$PYTHON_VERSION" pip
fi
conda activate "$ENV_DIR"

python -m pip install "$(pin 'tensorflow[and-cuda]' "$TENSORFLOW_VERSION")" "$(pin keras "$KERAS_VERSION")" \
    "$(pin numpy "$NUMPY_VERSION")" "$(pin scipy "$SCIPY_VERSION")" "$(pin scikit-learn "$SKLEARN_VERSION")" h5py pytest matplotlib  # matplotlib: imported by staintools, which is installed --no-deps below
python -m pip install -r "$CODE/ens_model_training/requirements.txt"
python -m pip install --no-deps staintools==2.1.2
python -m pip check || true   # report, do not fail: staintools is deliberately installed without deps
python -m pip freeze > "$ENV_DIR/nscc_env_freeze.txt"

mkdir -p "$KERAS_HOME"
cd "$CODE"
PYTHONPATH="$CODE" python - <<'EOF'
import json
import tensorflow as tf
from ens_model_training import ens_cache, ens_train
ens_train.keras3()  # Keras 3 is required, as in the Stage 2D environment
# Fetch NASNetLarge no-top weights now; build_model('imagenet') on a compute node then reads the cache.
tf.keras.applications.NASNetLarge(include_top=False, weights='imagenet', input_shape=(500, 500, 3))
print(json.dumps(dict(tensorflow=tf.__version__, keras=tf.keras.__version__,
                      built_cuda=tf.sysconfig.get_build_info().get('cuda_version'),
                      built_cudnn=tf.sysconfig.get_build_info().get('cudnn_version'),
                      software=ens_cache.software()), indent=1))
EOF
ls -l "$KERAS_HOME/models"
