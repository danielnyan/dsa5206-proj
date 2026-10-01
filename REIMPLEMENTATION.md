# Constrained binary reimplementation

This separate pipeline is **not an exact reproduction**. The released VAL1
patches have binary labels and no verified patient/slide grouping. Internal
patch-level validation may leak patient/slide characteristics. VAL2 stays locked
until final evaluation, with no development-time image reads.

## Frozen split

`python -B -m modern_pca.train_reimplementation --freeze-split`

Reads the checksum-authenticated Stage 2B VAL1 CSV, ranks each class by SHA-256
of UTF-8 `VAL1-internal-v1|42|<sample_id>` (ties: sample_id), and assigns the
first 18,602 benign and 10,562 tumour samples to internal validation. Remaining
74,407 benign and 42,248 tumour samples train. Each output retains Stage 2B
columns, canonical lexical ordering, and independently assigned zero-based
indices. Existing different split artifacts are rejected, never overwritten.
Artifacts are under `manifests/VAL1_internal_v1/`.

## Scientific procedure

NASNetLarge ImageNet backbone, 350x350 RGB, Flatten, Dense(256, ReLU),
Dense(2, softmax), order benign/tumour: **209,812,820 parameters**.
Raw reference SHA-256:
`1d31d575c21dcbcf172957b0783c37fcb07872d3129b3a7b3e86ede25ea0a1c1`.
Reuse the strict evaluator's RGB decode, Pillow LANCZOS resize, legacy
StainTools brightness, Macenko, float32/255 adapter without editing it.
Source files are read and their SHA-256 checked before preprocessing.
All preprocessing creates derived arrays; source images are never written.
Failures abort, with no replacement normalization or silent sample skipping.

Training independently flips each axis with probability 0.5 using stateless
seed 42 combined with sample identity and epoch. Each epoch orders training
samples by `SHA256(VAL1-epoch-v1|42|<epoch>|<sample_id>)`, ties sample_id.
No oversampling, class weights, rotation, extra warmup epoch, early stopping,
or validation-based model selection. Sparse categorical cross-entropy; Adam
beta1=.9, beta2=.999, epsilon=1e-7, no clipping/weight decay/AMSGrad.

Epochs 1–7 release from `normal_conv_1_17`, LR 1e-5; epochs 8–14 release from
14, 11, 9, 7, 5, 3, 1 respectively, LR 1e-6. Boundaries use backbone layer
order; all names are checked first. Adam resets per stage and retains state
within epochs 1–7. All BatchNorm layers stay frozen and run in inference mode
(explicit deviation from legacy training).

Physical batch 1, 2, or 4; effective batch 100. Accumulate **summed** sample-loss
gradients in float32, divide by the actual logical-group count on Adam update.
Flush the final partial group (55 training samples), never drop a sample.
Primary production model is always epoch 14, saved as `epoch14.keras` with a
checksum-bound sidecar. Production resumption is not implemented; the final
portable model is for inference. No production training has been authorized in
Stage 2D.

## Smoke benchmark

In a configured WSL GPU environment, from the repository:

Core tested package versions are pinned in `requirements-reimplementation-wsl.txt`.
The isolated environment is `dsa5206-reimplementation` (Python 3.11.16).
TensorFlow's supported GPU installation route uses its `and-cuda` extra in WSL2;
see the [official installation guide](https://www.tensorflow.org/install/pip).
SPAMS is supplied by the [spams-bin distribution](https://pypi.org/project/spams-bin/2.6.13/).
The run metadata records installed versions and preprocessing source hashes;
this environment is not asserted numerically identical to the historical stack.

```sh
python -B -m modern_pca.train_reimplementation --smoke-benchmark \
  --extracted '/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/data/zenodo/_3825933/extracted' \
  --output runs/stage2d_smoke
```

400 distinct deterministic **training-only** samples, interleaved equally by
class, including a native dimension variant of each class, are preprocessed
once into a checksum-bound derived cache. Each configuration starts from the
same seed and ImageNet weights in a fresh process. Stages 17 and 1 each test
physical batches 1, 2, 4, float32 first. Each runs 100 warmup samples plus 300
measured samples (four real Adam updates). No mock gradients or optimizer slots.
Float32 batch-1 OOM triggers separately labelled mixed-float16 experiments;
fixed loss scale 128, float32 accumulated unscaled gradients, abort on nonfinite
values. No silent precision or architecture fallback.

Synchronous TensorFlow execution and materialized return tensors ensure timers
include completed GPU work. Microbatch timing includes augmentation, transfer,
forward/backward, finite checks and accumulation; optimizer updates are timed
separately. Peak allocator memory is measured after warmup. Checkpoint testing
perturbs then restores model weights and real Adam state and compares every
tensor hash plus a prediction. Disposable checkpoint files are removed after a
successful integrity check; their checksums remain in the report.

Validation-time estimates use forward passes on these training samples only;
they do not read internal-validation or VAL2 images. Epoch estimates add measured
serial preprocessing cost. Fourteen-epoch estimates use endpoint-stage bounds
and exclude checkpoint I/O, intermediate-stage measurements, and thermal/IO
variation. These are estimates, not completed training results.

## Final evaluation (later)

`modern_pca.evaluate_reimplementation` requires an explicit cohort, manifest,
model, metadata, extracted root, and new output directory. VAL2 additionally
requires `--final-evaluation` and an epoch-14 production sidecar. Smoke weights
are rejected. Frozen model, zero warmup, no TTA/calibration/threshold search.
Tumour score is output[1], tumour iff score **>0.5**, ties benign. Full-precision
CSV probabilities and metrics include accuracy, precision, sensitivity,
specificity, F1, ROC-AUC, and confusion matrix [[TN,FP],[FN,TP]]. VAL2 accuracy
can be compared descriptively with the approximate publication value 96.7%,
explicitly as a constrained reimplementation with a different training task.

Console/file logging uses INFO by default, DEBUG optional; sample progress every
1000 images or 30 seconds. JSON timing separately records startup, model loading,
manifest validation, reference setup, preprocessing, materialized inference,
metrics, output, integrity checks and total. It reports per-image milliseconds
and throughput. Total excludes the final self-reporting timing JSON/completion
tail. Scientific metrics and performance results are separate artifacts.

For any later Windows CPU/WSL GPU comparison keep checkpoint/checksum, manifest,
preprocessing, input size, threshold, class order, dtype and batch identical.
Larger-batch measurements must be labelled performance-only experiments.

Run CPU tests first:
`python -B -m pytest tests/test_reimplementation.py tests/test_evaluate_paper.py tests/test_validation_manifests.py -q`.
