# DSA5206 Project — Prostate Cancer Pathology Reproducibility Study

This repository is for **DSA5206 — Advanced Topics in Data Science**.

The project studies the reproducibility of the prostate-cancer pathology work by **Tolkach Y. et al., “High-accuracy prostate cancer pathology using deep learning.”** The authors' original implementation is available at `https://github.com/gagarin37/deep_learning_pca`.

The original/legacy code is retained for reference. The current project also contains a separate modern constrained reimplementation and reproducibility pipeline.

---

## Current Status — Native-architecture 14-epoch checkpoint

This handover covers the **14-epoch constrained binary NASNetLarge checkpoint** (Stage 2H) and its prerequisites (Stages 2A–2G). The checkpoint reproduces the NASNetLarge architectural family and a staged training design, **not the authors' original three-class checkpoint, dataset or published experiment**.

| Milestone | Status |
|---|---|
| Dataset verification, frozen manifests, and deterministic VAL1 split (2A–2C) | PASS |
| GPU feasibility and cache/profiling work (2D–2E) | PASS / documented limitations |
| CPU/GPU numerical investigation (2F) | Material-equivalence GO; strict gradient parity FAIL |
| Full lossless VAL1 cache and trainer readiness (2G) | PASS |
| Native-architecture 14-epoch production run (2H) | **Completed on NUS Vanda** |

### Frozen production checkpoint

| Field | Verified record |
|---|---|
| Model | NASNetLarge, input 350×350×3, binary outputs (benign / tumour) |
| Completed training epochs | 14 |
| Original training patches | 116,655 (VAL1 internal training) |
| Internal validation patches | 29,164 (VAL1 internal validation) |
| Production platform | NUS Vanda, 2× NVIDIA A40 GPUs, TensorFlow/Keras |
| Model artifact | `epoch14.keras` (approximately 804 MiB; verify exact size on producer Vanda) |
| Model SHA-256 | `433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde` |
| Current producer Vanda path | `/scratch/e1536052/DSA5206/vanda_runs/nasnet_production_14epoch/epoch14.keras` |
| Shared checkpoint | GitHub Release asset, not a standard Git blob; insert actual release URL below after publishing |

**Release URL:** `TO_BE_FILLED_AFTER_UPLOAD`

The checkpoint can be used for research inference without rerunning 14 epochs. Full training reproduction additionally requires the original source images, frozen manifests, stain reference, lossless preprocessing cache (approximately 54 GB), training scripts and matching Python environment. These large data/cache assets are **not** stored in Git.

See [Team handover](#team-handover--running-the-14-epoch-model-on-vanda) below and `docs/VANDA_NATIVE_HANDOVER.md`.

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

VAL2 is an external cohort; it is not included in this handover.

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

At Stage 2E, the full VAL1 cache had **not yet been generated**. It was subsequently generated and fully verified in Stage 2G.

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

Strict CPU/GPU parity remained **FAIL** under the original predeclared
tolerances, reproducing exactly the same **8 of 47 gradient-tensor failures**.
The original tolerances were not relaxed.

Further diagnostics found:

- CPU repeatability: **bitwise exact**
- GPU repeatability: **bitwise exact**
- independent gradient-accumulation checks: **47/47 PASS**
- no accumulation inconsistency was found
- 10-update CPU/GPU prediction agreement remained aligned
- final comparison across 1,000 images: **1,000/1,000 predictions agreed**
- maximum probability difference: **0.0000174**

The observed discrepancies are therefore assessed as likely benign float32
CPU/GPU numerical variation rather than an implementation inconsistency.

**Scientific/material-equivalence assessment: GO for GPU training on numerical grounds.**

This does **not** claim that CPU and GPU training will remain bitwise identical
over a complete 14-epoch run. It establishes only that the measured short-run
numerical differences were not materially significant.

Detailed Stage 2F evidence:

- `reports/stage2f_numerical_parity.md`
- `reports/stage2f_numerical_parity.json`
- `reports/stage2f_checks.json`

Production training was not started **during Stage 2F**.
This statement describes only the historical milestone at that stage.

---

## Stage 2G — Full VAL1 Cache Generation and Trainer Readiness

Stage 2G generated and verified the complete lossless VAL1 preprocessing cache
used for the production-training path.

### Full cache result

| Measurement | Verified result |
|---|---:|
| Total VAL1 samples | 145,819 |
| Training samples | 116,655 |
| Internal-validation samples | 29,164 |
| NPY shards | 570 |
| Tensor payload | 53.589 GB |
| Metadata | 149.393 MB |
| Active cache size | ~53.738 GB |
| Full cache integrity | PASS |
| Source-image immutability | PASS |
| Trainer/cache integration | PASS |
| One-update GPU dry run | PASS |
| Checkpoint reload | PASS |
| Resume behavior | PASS |

The full cache is stored locally under:

```text
runs/stage2g_full_cache/cache
```

Because `runs/` is ignored by Git, the approximately 54 GB cache is **not**
uploaded to GitHub.

### Preprocessing parity

Exact online-versus-cache parity passed on:

- **590 representative samples**, covering both classes, both splits,
  source-dimension strata and every shard;
- **1,000 deterministic random samples**.

For both audits:

```text
PASS
maximum difference = 0
differing elements = 0
```

All 145,819 cached tensor checksums and corresponding original VAL1 JPEG
checksums were reverified.

### Trainer integration

The production trainer successfully traversed all 145,819 cached samples with
the expected order, labels and split assignments.

The bounded GPU dry run verified:

- physical batch size 4
- effective batch size 100
- float32 training
- TF32 disabled
- frozen BatchNormalization behavior
- gradient accumulation
- exactly one optimizer update
- validation pass
- checkpoint write and reload
- optimizer-state restoration
- resume cursor behavior

No production epoch was run **during Stage 2G**.

### Stage 2G performance

| Measurement | Result |
|---|---:|
| Full-cache loader throughput | 130.8 images/s |
| Revised 14-epoch training forecast | ~22.09 hours |
| Conservative serial-input estimate | ~26.48 hours |
| Recorded cache-generation time | at least 2.809 hours |
| Recorded generation throughput | at most 14.418 images/s |

The cache-generation timing is a **lower-bound timing claim** because one
generation session was interrupted. Correspondingly, the reported generation
throughput is an upper bound rather than an exact end-to-end throughput claim.

Stage 2G conclusion:

> **Full VAL1 cache: PASS. Trainer/cache integration: PASS. Production readiness: GO.**

This describes the position **at the end of Stage 2G**; Stage 2H production subsequently completed.
This statement describes only the historical milestone at that stage.

Detailed Stage 2G evidence:

- `reports/stage2g_full_cache.md`
- `reports/stage2g_full_cache.json`
- `reports/stage2g_checks.json`

---

## Stage 2H — Production 14-epoch NASNetLarge training (Vanda)

Production training completed for **14 epochs** on Vanda using the frozen VAL1 patch-level split and two NVIDIA A40 GPUs. The exported Keras checkpoint has the SHA-256 recorded above. The trained output head is **binary**, reflecting the released VAL1 labels rather than the authors' three-class training annotations.

The final Vanda model is external to Git. Place the verified `.keras` checkpoint into a GitHub Release so collaborators can retrieve it without expanding ordinary repository history. Preserve any available production training logs, final-model metadata, checkpoint manifest and run configuration as **small reviewed documentation/evidence files**, never the large optimizer checkpoint or preprocessing cache.

For numerical reproducibility, the existing accelerated Vanda training route uses distributed effective batches of 100; the earlier single-GPU local trainer uses a different microbatch and accumulation path. Their numerical identity has **not** been established. An exact 14-epoch rerun additionally requires the documented manifests, lossless cache, environment and original images.

Do **not** treat the internal VAL1 validation accuracy as an independent patient-level estimate: the VAL1 split is patch-level and underlying slide/patient provenance is missing.

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
    profile_validation.py
    numerical_parity.py
    diagnose_parity.py
    full_cache.py
    cached_training.py

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
    stage2f_numerical_parity.md
    stage2f_numerical_parity.json
    stage2f_checks.json
    stage2g_full_cache.md
    stage2g_full_cache.json
    stage2g_checks.json

tools/
    build_validation_manifests.py
    download_validation_data.ps1
    report_stage2f.py
    finish_stage2g.py
    profile_full_cache.py
    report_stage2g.py

tests/
    test_validation_manifests.py
    test_reimplementation.py
    test_stage2e.py
    test_diagnose_parity.py
    test_full_cache.py

4_WSI_pipeline/
    WSI_pipeline_v6/
```

Runtime outputs, downloaded datasets and large preprocessing caches are not intended to be committed to Git.

---

## Environment

The initial GPU development environment used WSL2; the completed 14-epoch production checkpoint was trained on NUS Vanda with 2× A40 GPUs.

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

Latest verified test status available in the supplied README (through Stage 2G):

```text
Stage 2E: 183 tests passed
Stage 2F: 190 tests passed
Stage 2G: 213 tests passed
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
- `reports/stage2f_numerical_parity.md` — CPU/GPU numerical-parity diagnosis and scientific assessment
- `reports/stage2f_numerical_parity.json` — machine-readable Stage 2F evidence
- `reports/stage2f_checks.json` — Stage 2F verification checks
- `reports/stage2g_full_cache.md` — full VAL1 cache, exact parity audits, trainer readiness and runtime forecast
- `reports/stage2g_full_cache.json` — machine-readable Stage 2G evidence
- `reports/stage2g_checks.json` — Stage 2G tests, protected-file hashes and Git checks

Large experimental artifacts under `runs/`, downloaded source data and the generated full preprocessing cache remain outside normal Git tracking.

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

## Team Handover — Running the 14-epoch Model on Vanda

For a teammate using a **different Vanda account**, the producer's `/scratch/e1536052/...` paths will not necessarily be readable. Each teammate must clone the reviewed Git commit, set up a compatible environment in their own scratch space, retrieve the separately released `epoch14.keras` asset, and verify its SHA-256 before inference.

### 1. Clone the reviewed source version

```bash
ssh YOUR_NUS_ID@vanda.nus.edu.sg
module load Python/3.11.5-GCCcore-13.2.0
mkdir -p "$HOME/scratch/DSA5206"  # adjust to your assigned scratch allocation
cd "$HOME/scratch/DSA5206"
git clone YOUR_REPO_SSH_URL dsa5206-proj
cd dsa5206-proj
git checkout YOUR_RELEASE_TAG
```

`YOUR_REPO_SSH_URL` and `YOUR_RELEASE_TAG` must be replaced with the actual GitHub repository and immutable release tag. Obtain the correct Python environment from `docs/VANDA_NATIVE_HANDOVER.md`; **do not** assume copying a virtual environment from another user's account will work.

### 2. Download the model asset

Download `epoch14.keras` from the release assets into your own scratch directory. Use GitHub CLI authentication or an authorized HTTPS download for a private repository. Do not assume an ordinary `wget` request to a private release asset is authenticated.

```bash
mkdir -p "$HOME/scratch/DSA5206/checkpoints"
# Example if gh is installed and authenticated:
gh release download YOUR_RELEASE_TAG -R OWNER/REPO \
  -p epoch14.keras -D "$HOME/scratch/DSA5206/checkpoints"
sha256sum "$HOME/scratch/DSA5206/checkpoints/epoch14.keras"
```

Expected hash: `433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde`.

### 3. Run research-only inference on a local patch

After verifying TensorFlow, Pillow, StainTools and the exact reference stain are usable on a Vanda compute node, run the separate single-image inference module in this handover:

```bash
python -m modern_pca.vanda_native_infer \
  --model "$HOME/scratch/DSA5206/checkpoints/epoch14.keras" \
  --image /PATH/TO/YOUR/PROSTATE_PATCH.jpg \
  --output /PATH/TO/OUTPUT/prediction.json
```

This checks the model checksum and executes the same legacy preprocessing (350×350 RGB Lanczos → brightness normalization → Macenko → `/255`). The input is a pathology image patch, **not an arbitrary whole-slide image**. This is for reproducibility research, **not clinical diagnosis**.

### 4. Reproducing all 14 training epochs is a separate workflow

An inference run needs the checkpoint, stain reference and images to be scored. A full fresh training run also needs the ~54 GB VAL1 cache, source training images, training manifests, two GPUs and a verified matching execution environment. Do not copy entire cache folders or rerun training unless required. Production PBS file location and full submission instructions must be verified against the tagged repository and institution-specific account settings before reuse.

See `docs/VANDA_NATIVE_HANDOVER.md` for the publisher checklist, release commands, environment test and source-code verification instructions.
