# DSA5206 Project Folder
This GitHub project is for DSA5206 - Advanced Topics in Data Science. The code was adapted from [High-accuracy prostate cancer pathology using deep learning by Tolkach Y. et al.](https://github.com/gagarin37/deep_learning_pca). Please refer to the original repository for more information regarding how to train, finetune and reuse the model. 

# Installation

Begin by checking out COLAB_RUNBOOK.md. The notebook runs on Colab, although it times out when you try to run full training. Might need to convert this into a Python script that could be run locally. Alternatively, you could run it as a Jupyter notebook on your laptop. 

Do note that full reproduction is not possible as Tolkach Y. used manually labelled TCGA data. Instead, this study splits the "validation data" published on Zenodo and uses that as the training / validation dataset instead. 

# Usage

Tolkach Y.'s original code is retained for legacy purposes. However, the main entry points are in `modern_pca/train.py` and `modern_pca/evaluate.py`. The options worth highlighting are as follows:  
- architecture, weights: an enumerable. This should be synced between train.py, evaluate.py and models.py. The model must be available at tf.keras.applications and have a publicly available checkpoint with ImageNet weights. Otherwise, consider refactoring the code if you want to load a custom model checkpoint.
- head: keep as "original". "gap" applies a global average pooling before the final classifier head, and is not the original architecture. 
- schedule: If set to "paper", does step-by-step upstream finetuning by unfreezing layers one by one
- stain-reference: Uses 4_WSI Pipeline, point to a reference image to perform Macenko normalisation such that the appearance of the stained slide will match the reference image. See stain.py for further details. There is also a fallback: if an image doesn't contain enough suitable stained pixels or the linear algebra fails, the code returns the brightness-standardized image instead of crashing.

The example PBS files that could be submitted are as follows:  

Data preparation: 
```bash
#!/bin/bash
#PBS -N dsa5206_prepare
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=12:00:00
#PBS -j oe

set -euo pipefail

cd "$PBS_O_WORKDIR"
source .venv/bin/activate

DATA_BASE="${SCRATCH:-$PBS_O_WORKDIR}/dsa5206_data"
DOWNLOADS="$DATA_BASE/downloads"
EXTRACTED="$DATA_BASE/extracted"
DATA_ROOT="$DATA_BASE/released_classes"

mkdir -p "$DOWNLOADS" "$EXTRACTED"

download_and_extract () {
    FILE="$1"
    EXPECTED="$2"

    curl -fL \
        --retry 8 \
        --retry-all-errors \
        -C - \
        -o "$DOWNLOADS/$FILE" \
        "https://zenodo.org/records/3825933/files/$FILE?download=1"

    echo "$EXPECTED  $DOWNLOADS/$FILE" | md5sum -c -

    DEST="$EXTRACTED/${FILE%.zip}"
    mkdir -p "$DEST"

    unzip -q -n "$DOWNLOADS/$FILE" -d "$DEST"
}

download_and_extract \
    val_dataset_1_norm.zip \
    1dda32f59996640d701b591f69353115

download_and_extract \
    val_dataset_1_tu.zip \
    35494acbfea3adbc689cc15f73a9116d

download_and_extract \
    val_dataset_2_norm.zip \
    029069ab405a76d42fcdefac32982989

download_and_extract \
    val_dataset_2_tu.zip \
    d72b78d364fbfed95d8f0d71d8f2d01e

python tools/prepare_zenodo.py \
    --archives-root "$EXTRACTED" \
    --output "$DATA_ROOT"

echo "Prepared dataset:"
echo "$DATA_ROOT"
du -sh "$DATA_BASE"
```

Submit training job: 
```
#!/bin/bash
#PBS -N dsa5206_nasnet
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=48:00:00
#PBS -j oe

set -euo pipefail

cd "$PBS_O_WORKDIR"
source .venv/bin/activate

DATA_BASE="${SCRATCH:-$PBS_O_WORKDIR}/dsa5206_data"
DATA_ROOT="$DATA_BASE/released_classes"

RUN_BASE="${SCRATCH:-$PBS_O_WORKDIR}/dsa5206_runs"
RUN_DIR="$RUN_BASE/released_full_nasnetlarge_original"

STAIN_REFERENCE="$PBS_O_WORKDIR/4_WSI_pipeline/WSI_pipeline_v6/images/standard_he_stain_small.jpg"

mkdir -p "$RUN_DIR"

echo "=== Environment ==="
hostname
date
python --version
nvidia-smi || true

python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY

echo "=== Training ==="

python -u -m modern_pca.train \
    --data-root "$DATA_ROOT" \
    --output "$RUN_DIR" \
    --architecture nasnetlarge \
    --head original \
    --weights imagenet \
    --schedule paper \
    --image-size 350 \
    --batch-size 2 \
    --seed 42 \
    --stain-reference "$STAIN_REFERENCE"

echo "=== Evaluation ==="

python -u -m modern_pca.evaluate \
    --model "$RUN_DIR/final.keras" \
    --data-root "$DATA_ROOT" \
    --output "$RUN_DIR/evaluation" \
    --image-size 350 \
    --batch-size 2 \
    --stain-reference "$STAIN_REFERENCE"

echo "=== Complete ==="
date
cat "$RUN_DIR/summary.json"
cat "$RUN_DIR/evaluation/evaluation.json"
```
