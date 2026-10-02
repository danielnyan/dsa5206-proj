# DSA5206 Project — Prostate Cancer Pathology Reproducibility Study

This repository is for **DSA5206 — Advanced Topics in Data Science**.

The project studies the reproducibility of the prostate-cancer pathology work by **Tolkach Y. et al., “High-accuracy prostate cancer pathology using deep learning.”** The authors' original implementation is available at `https://github.com/gagarin37/deep_learning_pca`.

The original/legacy code is retained for reference. The current project also contains a separate modern constrained reimplementation and reproducibility pipeline.

---

## Current Reproducibility Status

Exact reproduction of the published trained model is currently **not possible** because the original trained checkpoint and exact original training artifacts are unavailable. The authors were contacted and were unable to provide the missing artifacts.

The released Zenodo validation datasets are therefore used for a carefully controlled **constrained binary reimplementation**. Results from this repository must not be described as an exact reproduction of the published model.

| Stage | Description | Status |
|---|---|---|
| Stage 2A | Zenodo validation dataset verification | **PASS** |
| Stage 2B | Deterministic dataset manifests | **PASS** |
| Stage 2C | Reimplementation experimental design | **PASS** |
| Stage 2D | GPU feasibility and smoke benchmarking | **PASS** |
| Stage 2E | Performance profiling and preprocessing-cache validation | **COMPLETE** |
| Stage 2F | CPU/GPU numerical-gradient investigation | **COMPLETE** |
| Full VAL1 cache | One-time lossless preprocessing-cache generation | **NEXT** |
| Production training | 14-epoch constrained reimplementation | **NOT STARTED** |
| VAL2 evaluation | Final external evaluation | **NOT STARTED** |

> **Production training is currently NO-GO pending Stage 2F numerical-parity investigation.**

> **VAL2 has remained untouched during model development, profiling, performance optimization and hyperparameter decisions.**

---

## Reproducibility Limitation

The original study trained using manually annotated TCGA material and a three-class problem:

1. benign glandular tissue,
2. benign non-glandular tissue,
3. tumour.

The released VAL1 and VAL2 datasets contain only **benign** and **tumour** patch labels.

Therefore:

- the original three-class training problem cannot be reconstructed from the released validation patches;
- glandular/non-glandular labels are **not fabricated**;
- the constrained reimplementation uses a two-class output;
- results must be reported as a **constrained reimplementation**, not an exact reproduction.

The strict paper-faithful evaluator remains separate and should only be used with an authenticated original three-output checkpoint if one becomes available.

---

## Verified Dataset

The released validation data from **Zenodo record 3825933** was downloaded, checksum-verified, extracted and audited without modifying the source images.

| Cohort | Benign | Tumour | Total |
|---|---:|---:|---:|
| VAL1 | 93,009 | 52,810 | 145,819 |
| VAL2 | 24,471 | 9,632 | 34,103 |
| **Total** | **117,480** | **62,442** | **179,922** |

All four official ZIP archives passed MD5 verification:

| Archive | MD5 |
|---|---|
| `val_dataset_1_norm.zip` | `1dda32f59996640d701b591f69353115` |
| `val_dataset_1_tu.zip` | `35494acbfea3adbc689cc15f73a9116d` |
| `val_dataset_2_norm.zip` | `029069ab405a76d42fcdefac32982989` |
| `val_dataset_2_tu.zip` | `d72b78d364fbfed95d8f0d71d8f2d01e` |

Verification found:

- no partial downloads,
- no zero-byte files,
- no unreadable images,
- no case-only path collisions,
- no SHA-256 duplicate image contents within or across the requested groups,
- 179,922 readable RGB JPEGs,
- 1,837 legitimate native dimension variants retained unchanged.

VAL1 and VAL2 remain separate throughout the project.

See `DATASET.md` and `reports/dataset_verification.md` for details.

---

## Deterministic Dataset Manifests

Deterministic manifests were created for VAL1 and VAL2.

```text
VAL1_manifest.csv
c483f827985c241106942766062ad58b23c7ad86e6853ade51bb045bb5feff77

VAL2_manifest.csv
7b95ac02798e5971fc2722d3c37e8e7613bc1d020dee9a1a5d49ac939cf746d9
```

The manifests record sample identity, cohort, class, path, SHA-256, file size, dimensions and image mode.

See:

- `manifests/manifest_summary.md`
- `tools/build_validation_manifests.py`

---

## Frozen VAL1 Development Split

The released patch names and metadata do not provide reliable patient, WSI, case or source-region identifiers. A patient-independent split therefore cannot be reconstructed.

A deterministic class-stratified **patch-level** split is used:

| Split | Benign | Tumour | Total |
|---|---:|---:|---:|
| Training | 74,407 | 42,248 | 116,655 |
| Internal validation | 18,602 | 10,562 | 29,164 |

Frozen split SHA-256 fingerprints:

```text
VAL1 training
e6aaa451d54602fa19c86ea37b395043ed8fb0526122d6e780a1ecd410ebc3b8

VAL1 internal validation
cdcecc2362f175c4aa6163c2f6304071f11624bb1b57f4d0aef9f31e71055069
```

This internal validation split may contain patches originating from the same unknown patient or slide as the training set. It must therefore not be interpreted as patient-independent validation.

VAL2 remains the untouched external test cohort.

---

## Constrained Reimplementation Model

The primary reimplementation preserves the original NASNetLarge design as closely as possible while adapting the unavailable three-class supervision to the released two-class labels.

```text
Input: 350 x 350 x 3 RGB
        |
        v
NASNetLarge
ImageNet initialization
include_top=False
        |
        v
Flatten
        |
        v
Dense(256, ReLU)
        |
        v
Dense(2, Softmax)

Output 0 = benign
Output 1 = tumour
```

Actual total parameter count:

```text
209,812,820
```

Primary training configuration:

- ImageNet-initialized NASNetLarge
- 350 x 350 RGB input
- legacy-style brightness standardization
- StainTools Macenko normalization
- float32 `/255` scaling
- horizontal and vertical flip augmentation
- sparse categorical cross-entropy
- Adam optimizer
- staged 14-epoch unfreezing schedule
- frozen BatchNormalization layers in inference mode
- physical batch size 4
- effective batch size 100 using gradient accumulation
- seed 42
- primary checkpoint = epoch 14

The current reimplementation entry points are:

- `modern_pca/train_reimplementation.py`
- `modern_pca/evaluate_reimplementation.py`
- `modern_pca/reimplementation.py`

The strict evaluator for an authenticated original three-output checkpoint is:

- `modern_pca/evaluate_paper.py`

Do not repurpose `evaluate_paper.py` for the binary constrained reimplementation.

---

## Preprocessing

The deterministic preprocessing sequence is:

```text
source JPEG
    |
    v
PIL RGB decode
    |
    v
Pillow LANCZOS resize to 350 x 350
    |
    v
brightness/luminosity standardization
    |
    v
StainTools Macenko normalization
    |
    v
uint8 normalized image
    |
    v
float32 / 255
```

Stain reference:

```text
4_WSI_pipeline/WSI_pipeline_v6/images/standard_he_stain_small.jpg
```

Reference SHA-256:

```text
1d31d575c21dcbcf172957b0783c37fcb07872d3129b3a7b3e86ede25ea0a1c1
```

Training-time horizontal and vertical flips are applied **after** deterministic preprocessing and are not included in the preprocessing cache.

Source JPEGs are never overwritten.

---

## Stage 2D — GPU Feasibility

Stage 2D verified that the constrained NASNetLarge model can run on:

```text
NVIDIA GeForce RTX 4080 Laptop GPU
~12 GB VRAM
WSL2
TensorFlow 2.20.0
Keras 3.15.1
```

All six tested float32 GPU configurations passed finite-gradient, weight-update, frozen-layer and checkpoint-restoration checks.

Physical batch 4 was selected.

| Physical batch | First-stage images/s | Deepest-stage images/s | Deepest peak TF allocation |
|---:|---:|---:|---:|
| 1 | 12.00 | 5.05 | 4.87 GiB |
| 2 | 18.08 | 8.12 | 5.22 GiB |
| **4** | **25.41** | **11.96** | **5.82 GiB** |

The initial serial end-to-end estimate was approximately **166.7–177 hours** for 14 epochs because deterministic Macenko preprocessing was being repeated every epoch.

See `reports/stage2d_implementation.md`.

---

## Stage 2E — Performance Profiling and Cache Validation

Stage 2E identified preprocessing, particularly Macenko normalization, as the main runtime bottleneck.

| Measurement | Result |
|---|---:|
| Macenko share of timed preprocessing | 85.9% |
| Online preprocessing/input | 4.82 images/s |
| Cached loader | 770.6 images/s |
| First-stage training, online | 3.99 images/s |
| First-stage training, cached | 37.52 images/s |
| Deepest-stage cached training | 13.82 images/s |
| First-stage GPU utilization, online | 10.2% |
| First-stage GPU utilization, cached | 66.2% |
| Recommended cache | Uncompressed uint8 NPY shards |
| Estimated full VAL1 cache | ~53.589 GB + ~167 MB metadata |
| Estimated one-time cache generation | ~8.87 hours |
| Revised 14-epoch training estimate | ~21.97 hours |
| Estimated cache + training time | ~30.84 hours |
| Conditional planning range | ~30.84–40.88 hours |

### Cache parity

The lossless preprocessing cache reproduced online preprocessing exactly:

```text
PASS
numpy.array_equal = True
max absolute difference = 0
differing elements = 0
```

This means deterministic preprocessing can be performed once and reused across epochs without changing the cached image values.

The full VAL1 cache has **not yet been generated**.

---

## CPU/GPU Numerical-Parity Status

Stage 2E also compared the same modern implementation on CPU and GPU.

Results:

- forward inference comparison: **PASS**
- post-update weight comparison: **PASS**
- post-update prediction comparison: **PASS**
- strict gradient comparison: **FAIL**
- 8 of 47 gradient tensors exceeded the predeclared numerical tolerances

The tolerances were **not relaxed** after observing the result.

This does not automatically prove that GPU training is scientifically invalid, because CPU and GPU floating-point accumulation can differ. However, the discrepancy must be investigated before production training.

## Stage 2F — CPU/GPU Numerical-Parity Investigation

Strict CPU/GPU parity remained FAIL under the original predeclared
tolerances, reproducing the same 8 of 47 gradient-tensor failures.

Further diagnostics found:

- CPU repeatability: bitwise exact
- GPU repeatability: bitwise exact
- gradient accumulation checks: 47/47 PASS
- 10-update CPU/GPU prediction agreement: 100%
- final comparison: 1,000/1,000 predictions agreed
- maximum probability difference: 0.0000174

The differences are assessed as likely benign float32 CPU/GPU numerical
variation rather than an implementation inconsistency.

Scientific/material-equivalence assessment: GO for GPU training on
numerical grounds.

Next step: generate and verify the full VAL1 preprocessing cache.

Production training has not started.
VAL2 remains untouched.

---

## Repository Layout

Important project components:

```text
modern_pca/
    evaluate_paper.py
    evaluate_reimplementation.py
    reimplementation.py
    train_reimplementation.py
    preprocessing_cache.py
    profile_reimplementation.py
    numerical_parity.py

manifests/
    VAL1_manifest.csv
    VAL2_manifest.csv
    VAL1_internal_v1/
    manifest_summary.md
    manifest_summary.json

reports/
    dataset_verification.md
    dataset_verification.json
    stage2d_implementation.md
    stage2d_implementation.json
    stage2e_performance.md
    stage2e_performance.json
    stage2e_checks.json

tools/
    build_validation_manifests.py
    download_validation_data.ps1

tests/
    test_validation_manifests.py
    test_reimplementation.py

4_WSI_pipeline/
    WSI_pipeline_v6/
```

Runtime outputs, downloaded datasets and large preprocessing caches are not intended to be committed to Git.

---

## Environment

The current GPU reimplementation environment is based on WSL2.

Important verified software from the Stage 2D environment includes:

```text
Python 3.11
TensorFlow 2.20.0
Keras 3.15.1
Pillow 10.4.0
NumPy 1.26.4
scikit-learn 1.5.2
StainTools 2.1.2
```

See `requirements-reimplementation-wsl.txt` for the pinned reimplementation environment.

---

## Tests

Run the repository test suite before production work:

```bash
python -m pytest -q
```

At completion of Stage 2E:

```text
183 tests passed
git diff --check passed
```

---

## Dataset Download

The official Zenodo validation archives are available from record 3825933.

A helper script is provided:

```text
tools/download_validation_data.ps1
```

Dataset details, checksums, expected layout and reproducibility rules are documented in `DATASET.md`.

Do not commit downloaded datasets to Git.

---

## Reproducibility Documentation

Use this README as the project landing page. Detailed evidence and implementation notes are maintained separately:

- `DATASET.md` — dataset source, download and verification
- `REIMPLEMENTATION.md` — constrained reimplementation design and usage
- `manifests/manifest_summary.md` — deterministic dataset manifests
- `reports/dataset_verification.md` — Stage 2A dataset verification
- `reports/stage2d_implementation.md` — GPU feasibility and smoke benchmark
- `reports/stage2e_performance.md` — profiling, preprocessing cache and CPU/GPU numerical-parity results
- `reports/stage2e_checks.json` — Stage 2E verification checks

Large experimental artifacts under `runs/`, downloaded source data and future full preprocessing caches should remain outside normal Git tracking.

---

## Legacy Code and Earlier Workflow

The authors' original code is retained for legacy/reference purposes.

Earlier project work used:

- `modern_pca/train.py`
- `modern_pca/evaluate.py`
- `COLAB_RUNBOOK.md`
- Colab/PBS examples

These files remain useful historical references but are **not the primary entry points for the current constrained reimplementation**.

The original repository should still be consulted when comparing legacy training, fine-tuning and inference behavior:

`https://github.com/gagarin37/deep_learning_pca`

---

## Current Next Step

```text
Stage 2F
CPU/GPU numerical-gradient investigation
        |
        v
If scientifically acceptable
        |
        v
Generate and verify full VAL1 preprocessing cache once
        |
        v
Short production dry run
        |
        v
14-epoch production training
        |
        v
Freeze epoch-14 checkpoint
        |
        v
Final untouched VAL2 evaluation
```

Do not begin production training or use VAL2 for development until the Stage 2F gate has been resolved.
