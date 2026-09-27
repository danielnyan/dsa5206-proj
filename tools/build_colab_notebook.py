from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "Prostate_Pathology_Colab_Runbook.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


cells = [
    md("""# Tolkach prostate-pathology: full Colab execution runbook

This notebook runs the modernized TensorFlow/Keras pipeline from uploaded source.
It executes real NASNetLarge training, validation, model reload, C1/C8 evaluation,
and separate CPU/GPU timing. Read the reproduction boundary before claiming a
paper reproduction."""),
    md("""## Public-artifact boundary

The author repository contains no trained weights. Zenodo 3825933 contains the
two tumour/benign validation cohorts (15.3 GB), not the annotated training tiles.
The TCGA slides are public but the author masks are missing; Gleason validation
images and clinical labels are also request-only. `released_full` is therefore a
real full-data retraining experiment on released patches, **not** independent
reproduction of the published result."""),
    code("""# User configuration
RUN_MODE = "examples"  # "examples" or "released_full"
RUN_PAPER_SCHEDULE = False  # True executes all 14 original epochs
ARCHITECTURE = "nasnetlarge"
HEAD = "original"  # exact paper head; "gap" is an explicit alternative
ALTERNATIVE_ARCHITECTURES = []  # e.g. ["resnet50", "efficientnetv2b0"]
BATCH_SIZE = 2
IMAGE_SIZE = 350
SEED = 42
SOURCE_ZIP = "deep_learning_pca_modern.zip"
"""),
    md("""## 1. Upload and unpack the patched repository

Upload the supplied ZIP to `/content`; no personal GitHub repository is needed."""),
    code("""from pathlib import Path
import os, shutil, subprocess, sys, zipfile

CONTENT = Path("/content")
archive = CONTENT / SOURCE_ZIP
if not archive.exists():
    from google.colab import files
    uploaded = files.upload()
    archive = CONTENT / next(name for name in uploaded if name.endswith(".zip"))
PROJECT = CONTENT / "deep_learning_pca"
if PROJECT.exists():
    shutil.rmtree(PROJECT)
with zipfile.ZipFile(archive) as stream:
    stream.extractall(CONTENT)
candidate = next(path for path in CONTENT.iterdir() if path.is_dir() and (path / "modern_pca").exists())
if candidate != PROJECT:
    candidate.rename(PROJECT)
os.chdir(PROJECT)
print("PROJECT", PROJECT)
"""),
    code("""# Current Colab TensorFlow is retained when it satisfies the modern lower bound.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "libopenslide0"], check=True)
"""),
    code("""import json, hashlib, platform, time
import tensorflow as tf

print("Python", platform.python_version())
print("TensorFlow", tf.__version__)
print("GPUs", tf.config.list_physical_devices("GPU"))
if RUN_MODE == "released_full":
    usage = shutil.disk_usage(CONTENT)
    assert usage.free >= 40 * 2**30, f"Need at least 40 GiB free; found {usage.free / 2**30:.1f}"
"""),
    md("""## 2. Prepare real histology data

`examples` uses the 30 author-supplied stain-normalized training patches for a
fast but genuine end-to-end execution. `released_full` downloads every exact
Zenodo validation archive using resumable `curl`, verifies MD5, and indexes all
images. The latter split is newly generated and must not be called the published
independent validation."""),
    code("""ZENODO = {
    "val_dataset_1_norm.zip": ("1dda32f59996640d701b591f69353115", 7.4),
    "val_dataset_1_tu.zip": ("35494acbfea3adbc689cc15f73a9116d", 4.9),
    "val_dataset_2_norm.zip": ("029069ab405a76d42fcdefac32982989", 2.0),
    "val_dataset_2_tu.zip": ("d72b78d364fbfed95d8f0d71d8f2d01e", 0.94),
}

DATA = CONTENT / "prostate_data"
DATA.mkdir(exist_ok=True)
if RUN_MODE == "released_full":
    downloads, extracted = DATA / "downloads", DATA / "extracted"
    downloads.mkdir(exist_ok=True); extracted.mkdir(exist_ok=True)
    for filename, (expected_md5, _) in ZENODO.items():
        target = downloads / filename
        url = f"https://zenodo.org/records/3825933/files/{filename}?download=1"
        subprocess.run(["curl", "-fL", "--retry", "8", "--retry-all-errors", "-C", "-", "-o", str(target), url], check=True)
        checksum = hashlib.md5()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 2**20), b""):
                checksum.update(chunk)
        digest = checksum.hexdigest()
        assert digest == expected_md5, (filename, digest, expected_md5)
        destination = extracted / filename.removesuffix(".zip")
        destination.mkdir(exist_ok=True)
        subprocess.run(["unzip", "-q", "-n", str(target), "-d", str(destination)], check=True)
    DATA_ROOT = DATA / "released_classes"
    subprocess.run([sys.executable, "tools/prepare_zenodo.py", "--archives-root", str(extracted), "--output", str(DATA_ROOT)], check=True)
else:
    DATA_ROOT = DATA / "example_classes"
    subprocess.run([sys.executable, "tools/prepare_examples.py", "--repo", str(PROJECT), "--output", str(DATA_ROOT)], check=True)
print("DATA_ROOT", DATA_ROOT)
STAIN_REFERENCE = PROJECT / "4_WSI_pipeline" / "WSI_pipeline_v6" / "images" / "standard_he_stain_small.jpg"
STAIN_ARGS = [] if RUN_MODE == "examples" else ["--stain-reference", str(STAIN_REFERENCE)]
"""),
    md("""## 3. Exact-model CPU/GPU benchmark

Each subprocess constructs the full selected model and performs forward and
backward updates on real histology images. CPU and GPU are separate processes so
TensorFlow device placement is unambiguous."""),
    code("""benchmark_base = [
    sys.executable, "-m", "modern_pca.benchmark", "--data-root", str(DATA_ROOT),
    "--architecture", ARCHITECTURE, "--head", HEAD, "--image-size", str(IMAGE_SIZE),
    "--batch-size", "1", "--inference-steps", "3", "--train-steps", "1",
] + STAIN_ARGS
cpu_env = os.environ.copy(); cpu_env["CUDA_VISIBLE_DEVICES"] = "-1"
cpu_run = subprocess.run(benchmark_base, check=True, text=True, capture_output=True, env=cpu_env)
print("CPU")
print(cpu_run.stdout)
gpu_run = None
if tf.config.list_physical_devices("GPU"):
    gpu_run = subprocess.run(benchmark_base, check=True, text=True, capture_output=True)
    print("GPU")
    print(gpu_run.stdout)
"""),
    md("""## 4. Train, validate, save, and reload

The decision schedule runs one complete epoch. The paper schedule performs the
released progressive unfreezing recipe: 7 + seven 1-epoch stages. No batch or
epoch is synthetically mocked."""),
    code("""RUN_DIR = CONTENT / "prostate_runs" / f"{RUN_MODE}_{ARCHITECTURE}_{HEAD}"
schedule = "paper" if RUN_PAPER_SCHEDULE else "decision"
train_command = [
    sys.executable, "-m", "modern_pca.train", "--data-root", str(DATA_ROOT),
    "--output", str(RUN_DIR), "--architecture", ARCHITECTURE, "--head", HEAD,
    "--weights", "imagenet", "--schedule", schedule,
    "--image-size", str(IMAGE_SIZE), "--batch-size", str(BATCH_SIZE), "--seed", str(SEED),
] + STAIN_ARGS
subprocess.run(train_command, check=True)
summary = json.loads((RUN_DIR / "summary.json").read_text())
print(json.dumps(summary, indent=2))
import modern_pca  # registers the serializable compatibility layers
reloaded = tf.keras.models.load_model(RUN_DIR / "final.keras")
print("RELOADED_PARAMETERS", reloaded.count_params())
"""),
    md("""## 5. Complete held-out C1 and C8 evaluation"""),
    code("""subprocess.run([
    sys.executable, "-m", "modern_pca.evaluate",
    "--model", str(RUN_DIR / "final.keras"), "--data-root", str(DATA_ROOT),
    "--output", str(RUN_DIR / "evaluation"), "--image-size", str(IMAGE_SIZE),
    "--batch-size", str(BATCH_SIZE),
] + STAIN_ARGS, check=True)
evaluation = json.loads((RUN_DIR / "evaluation" / "evaluation.json").read_text())
print(json.dumps(evaluation, indent=2))
"""),
    md("""## 6. Full-schedule runtime projection and optional alternatives

The projection is derived from the measured real epoch. GPU memory and I/O may
make later unfrozen stages slower, so use it as a lower-bound planning estimate."""),
    code("""completed_epochs = 14 if RUN_PAPER_SCHEDULE else 1
seconds_per_epoch = summary["elapsed_seconds"] / completed_epochs
print("MEASURED_SECONDS_PER_EPOCH", seconds_per_epoch)
print("PROJECTED_14_EPOCH_HOURS", seconds_per_epoch * 14 / 3600)

alternative_summaries = {}
for architecture in ALTERNATIVE_ARCHITECTURES:
    alt_dir = CONTENT / "prostate_runs" / f"{RUN_MODE}_{architecture}_gap"
    command = [
        sys.executable, "-m", "modern_pca.train", "--data-root", str(DATA_ROOT),
        "--output", str(alt_dir), "--architecture", architecture, "--head", "gap",
        "--weights", "imagenet", "--schedule", "decision", "--image-size", str(IMAGE_SIZE),
        "--batch-size", str(BATCH_SIZE), "--seed", str(SEED),
    ] + STAIN_ARGS
    subprocess.run(command, check=True)
    alternative_summaries[architecture] = json.loads((alt_dir / "summary.json").read_text())
print(json.dumps(alternative_summaries, indent=2))
"""),
    md("""## 7. Package results

Download this archive before the Colab runtime is recycled."""),
    code("""results_archive = shutil.make_archive(str(CONTENT / "prostate_benchmark_results"), "zip", CONTENT / "prostate_runs")
print("RESULTS_ARCHIVE", results_archive)
try:
    from google.colab import files
    files.download(results_archive)
except ImportError:
    pass
"""),
    md("""## Interpretation gate

- Passing the notebook proves the modern runtime and available-data pipeline.
- `released_full` measures whether full released-patch retraining and alternatives
  are affordable on Colab.
- It does **not** recover the missing author training annotations/models or the
  requested Gleason/clinical data. Exact headline-result replication remains
  blocked unless those artifacts are obtained from the authors."""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.parent.mkdir(exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1))
print(OUTPUT)
