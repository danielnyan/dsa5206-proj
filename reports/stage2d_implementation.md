# Stage 2D implementation and GPU smoke results

**PASS: all six float32 GPU configurations. GO for GPU feasibility at physical batch 4, effective batch 100.**
No production training was started. VAL2 images were not accessed. Source images and the strict paper evaluator were not modified.

This is a constrained binary reimplementation, not an exact reproduction. Patient/slide grouping remains unavailable; internal patch validation can leak group characteristics.

## Frozen VAL1 split

| Split | Benign | Tumour | Total |
|---|---:|---:|---:|
| Training | 74,407 | 42,248 | 116,655 |
| Internal validation | 18,602 | 10,562 | 29,164 |

CSV hashes (also reverified after benchmarking):

- VAL1_internal_validation.csv: `cdcecc2362f175c4aa6163c2f6304071f11624bb1b57f4d0aef9f31e71055069`
- VAL1_train.csv: `e6aaa451d54602fa19c86ea37b395043ed8fb0526122d6e780a1ecd410ebc3b8`

Counts, unique identities/indices, class mapping, complete VAL1 coverage and non-overlap: PASS.

## Model and unfreezing

Actual model parameters: **209,812,820**. NASNetLarge + Flatten + Dense(256, ReLU) + Dense(2, softmax), 350x350 RGB, ImageNet initialization. All BN frozen/inference mode.

| Epochs | Boundary | Trainable parameters | Learning rate |
|---|---|---:|---:|
| 1-7 | normal_conv_1_17 | 142,263,170 | 1e-05 |
| 8 | normal_conv_1_14 | 172,378,178 | 1e-06 |
| 9 | normal_conv_1_11 | 193,794,818 | 1e-06 |
| 10 | normal_conv_1_9 | 198,865,730 | 1e-06 |
| 11 | normal_conv_1_7 | 203,485,058 | 1e-06 |
| 12 | normal_conv_1_5 | 206,213,042 | 1e-06 |
| 13 | normal_conv_1_3 | 207,506,642 | 1e-06 |
| 14 | normal_conv_1_1 | 208,800,242 | 1e-06 |

ImageNet file SHA-256: `c4e84e674635001db3548bad5cc652e7dd314a03f36612b2a1d82f45bef69a6a`.
The Keras expected MD5 also passed. Full installed package pins: ../runs/stage2d_smoke/pip-freeze.txt.

## GPU smoke measurements

WSL2, NVIDIA GeForce RTX 4080 Laptop GPU, 12,282 MiB hardware VRAM; TensorFlow 2.20.0, Keras 3.15.1. GPU recognized; float32, TF32 disabled, deterministic operations.
400 distinct VAL1-training images: 200 benign, 200 tumour, 3 native dimension variants. Identical sample cache across configurations. One warmup logical batch plus three measured logical batches per configuration, 100 samples each.
Real loss, backpropagation, Adam slots and float32 accumulation. Each configuration passed finite loss/gradients, actual weight updates, frozen-weight stability and full model/Adam checkpoint restoration.

| Stage | Physical batch | Training images/s | Microbatch median / p95 (ms) | Adam update median (ms) | Peak TF allocation (GiB) | Status |
|---|---:|---:|---:|---:|---:|---|
| normal_conv_1_17 | 1 | 12.00 | 81.37 / 90.10 | 57.65 | 3.53 | PASS |
| normal_conv_1_17 | 2 | 18.08 | 106.99 / 125.80 | 55.77 | 3.59 | PASS |
| normal_conv_1_17 | 4 | 25.41 | 152.24 / 167.33 | 60.42 | 3.60 | PASS |
| normal_conv_1_1 | 1 | 5.05 | 193.67 / 218.32 | 182.91 | 4.87 | PASS |
| normal_conv_1_1 | 2 | 8.12 | 242.61 / 254.82 | 157.46 | 5.22 | PASS |
| normal_conv_1_1 | 4 | 11.96 | 325.26 / 352.77 | 151.85 | 5.82 | PASS |

Peak values describe TensorFlow allocations, not all driver/display memory or allocator reservations. Warmup peaks are also retained in JSON. No OOM occurred, so no mixed-precision fallback was needed.
Timings use perf_counter, synchronous TF execution and materialized compiled-function returns. Microbatches include augmentation/transfers/finite checks/accumulation. Checkpoint I/O and integrity hashing are outside measured training throughput.
Serial preprocessing/cache materialization: 91.23 s for 400 images (4.384 images/s).

## Runtime planning at recommended batch 4

| Stage | Training per epoch | Internal validation | Combined |
|---|---:|---:|---:|
| normal_conv_1_17 | 8.67 h | 3.24 h | 11.91 h |
| normal_conv_1_1 | 10.10 h | 3.28 h | 13.38 h |

Endpoint-based 14-epoch planning range: **166.7-177.0 hours (6.95-7.37 days)**.
This is an extrapolation, not a guaranteed bound. Intermediate stages, full-cohort stain failures, long-run thermal behavior and checkpoint I/O were not measured. Validation timing uses forward passes on training-only smoke images, never held-out images.
The current production pipeline repeats serial Macenko preprocessing each epoch and uses eager validation calls. These dominate practical runtime. A separately verified performance pass (for example lossless derived preprocessing caching and compiled inference) is advisable before committing to a week-long run; it must preserve scientific inputs/logic.
Only these 400 images were stain-preflighted. A later normalization failure aborts; samples are not silently dropped.

## Files and verification

- New pipeline: modern_pca/train_reimplementation.py, modern_pca/evaluate_reimplementation.py, modern_pca/reimplementation.py.
- Tests: tests/test_reimplementation.py. Final WSL CPU suite: **169 passed**. Earlier Windows suite: 165 passed, 1 skipped.
- Frozen split: manifests/VAL1_internal_v1/VAL1_train.csv, VAL1_internal_validation.csv, VAL1_split_summary.json.
- Usage/design: REIMPLEMENTATION.md; core WSL pins: requirements-reimplementation-wsl.txt.
- Results: reports/stage2d_implementation.json and this report; runs/stage2d_smoke/ contains JSON results, logs, exact sample CSV, derived cache, package pins, and executed source snapshots.
- Disposable smoke checkpoints were tested then removed; checksums retained. No production checkpoint exists.
- git diff --check: **PASS**. No commit or push.

After the benchmark, metadata lookup was corrected to Keras' actual lowercase cache filename, dependency hashes were added, and output/dataset overlap and warning-logging safeguards were hardened. The exact benchmark source is preserved. AST comparisons confirm numerical preprocessing/training/benchmark functions are unchanged. Each run has an explicitly timestamped post-run ImageNet checksum supplement.

Git status (existing Stage 2A/2B artifacts are included):

```text
?? REIMPLEMENTATION.md
?? manifests/
?? modern_pca/evaluate_reimplementation.py
?? modern_pca/reimplementation.py
?? modern_pca/train_reimplementation.py
?? reports/
?? requirements-reimplementation-wsl.txt
?? runs/
?? tests/test_reimplementation.py
?? tests/test_validation_manifests.py
?? tools/build_validation_manifests.py
```
