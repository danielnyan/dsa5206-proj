# Stage 2E: bounded performance and numerical-parity report

Generated 2026-10-01T22:44:39.955676+00:00. Raw evidence: `runs\stage2e_profile`.

**Cache parity: PASS. CPU/GPU parity: FAIL.**
Full-cache recommendation: **GO (recommendation only)**. Production launch: **NO-GO**.

No production training or full cache was started. No VAL2 images were accessed. Source JPEGs, strict original-checkpoint evaluator, model/training configuration and frozen split were not changed.

## Scope and controls

1000 deterministic VAL1-training samples: {'benign': 638, 'tumour': 362}; native dimensions: {'610x612': 990, '610x611': 2, '610x609': 2, '609x612': 4, '610x610': 2}. Exact cache parity re-ran the unchanged legacy helper on 64 representative samples, including both labels and all selected dimension strata. All 1000 source hashes were rechecked. Four formats and every optimized loader configuration preserved every tested value.

All training probes use the same saved initialization, ImageNet weights, 350x350 input, 209,812,820 parameters, float32 with TF32 disabled, frozen BN, physical batch 4 and effective batch 100. One logical batch warms up; three are measured. These disposable short runs are profiling, not fitted models for scientific evaluation.

## Preprocessing

| Component | Total s | ms/image | images/s | % instrumented wall |
|---|---:|---:|---:|---:|
| augmentation | 4.9662 | 4.9662 | 201.36 | 2.17 |
| batching | 0.6419 | 0.6419 | 1557.76 | 0.28 |
| brightness | 2.6633 | 2.6633 | 375.48 | 1.16 |
| decode_rgb | 2.1906 | 2.1906 | 456.49 | 0.96 |
| file_read | 7.5483 | 7.5483 | 132.48 | 3.30 |
| float32_conversion | 0.2895 | 0.2895 | 3454.26 | 0.13 |
| host_to_GPU_transfer | 3.5706 | 3.5706 | 280.07 | 1.56 |
| integrity | 0.0796 | 0.0796 | 12560.21 | 0.03 |
| macenko | 104.9748 | 104.9748 | 9.53 | 45.87 |
| resize | 4.7672 | 4.7672 | 209.77 | 2.08 |
| scaling | 0.4376 | 0.4376 | 2285.39 | 0.19 |

Instrumented wall: 228.86 s. Unattributed path/stat safety checks, cache writing, hashing, allocation and logging overhead: 96.73 s. These residual costs were not separated and are not assigned to Macenko. A-E operations alone: 8.187 images/s; cache-generation estimate including observed residual: 4.567 images/s. F-J are excluded from the one-time cache cost because the stored tensor precedes those operations.

Macenko is the largest measured preprocessing operation and runs on CPU. Compare endpoint online/cached GPU activity below to assess input starvation. The instrumented serial pass also includes small H2D probes; its utilization is not a Macenko-only measurement.

Macenko accounts for 85.94% of timed A-E preprocessing operations. Separately, online input preparation during training (including source-path checks, scaling, flips and batching) measured 4.82 images/s at the first stage and 5.22 images/s at the deepest stage; these rates exclude the subsequent training microbatch.

Serial preprocessing utilization: `{"available": true, "low_gpu_definition": "sampled utilization <10%; proxy for idle, not measured GPU stall time", "low_gpu_utilization_fraction": 1.0, "mean": {"cpu_process_percent_one_core": 1616.4226359434751, "cpu_system_percent": 73.64668178759284, "gpu_percent": 0.4301994301994302}, "peak_gpu_memory_mib": 202.0, "samples": 351}`.

All timings use `time.perf_counter()`. GPU tensors and compiled outputs are materialized with `.numpy()` before stopping timers; H2D includes a scalar reduction and synchronization. Backward diagnostics materialize a gradient norm. Timings therefore include completion, not just GPU launch latency.

## Cache formats

| Format | Subset bytes | Sequential images/s | Shuffled images/s | Exact |
|---|---:|---:|---:|---|
| individual_npy | 367,628,000 | 179.8 | 198.0 | True |
| raw_mmap | 367,500,000 | 36175.3 | 33302.0 | True |
| sharded_npy | 367,500,512 | 32017.5 | 34425.7 | True |
| tfrecord_indexed | 367,516,000 | 377.1 | 367.3 | True |

Read rates are medians of three warm-filesystem-cache passes copying full arrays; they are not full-dataset cold-disk rates. TFRecord sequential and shuffled figures use the same indexed raw-record reader; TensorFlow native CRC-checked decoding was separately round-trip verified. They do not benchmark an optimized streaming TFRecord pipeline.

**Recommend 256-image NPY shards.** They retain shape/dtype headers, allow direct read-only mmap and random sample access, and avoid 145,819 small-file opens. Individual NPY is simple to resume but costly on the WSL/NTFS boundary. Raw mmap is similarly fast but requires an external schema and offsets. TFRecord has native TensorFlow streaming support and CRCs, but efficient arbitrary shuffled access requires an index or changes to loading order; the indexed prototype is slower here.

Use one writer, temporary shard + fsync + atomic rename, then a committed SHA-256 sidecar. Resume only matching context and verified committed shards. Each process constructs its own read-only mmap handles; parallel reads do not mutate shared arrays. Per-image source and tensor hashes permit individual verification. Production conversion needs a train/validation-aware extension; the present builder deliberately caps at 1024 training samples.

Metadata records sample ID, relative source path, JPEG/tensor SHA-256, shape, dtype, class, split, shard and index. Global context records source/split/reference hashes, implementation fingerprints, resize/brightness/Macenko settings, dependency versions and backend identities. Augmentation is never cached.

Full VAL1 tensor payload: **53,588,482,500 bytes**. Separate train/validation 256-image NPY shards: **53,588,555,460 bytes** (53.589 GB; 49.908 GiB), 570 shards. Exact container total excludes JSON metadata, filesystem allocation and temporary workspace. Preserve source JPEGs separately; budget additional space for hashes/metadata and one temporary shard.

Subset JSON metadata occupies 1,146,286 bytes. A linear projection is approximately 167.2 MB for full VAL1; this is an estimate, not an exact future metadata-file size.

## Input optimization checks

| Parallel readers | Prefetch | Device prefetch | images/s including verification | Exact values/order/labels/flips |
|---:|---:|---|---:|---|
| 1 | 0 | False | 809.73 | True |
| 1 | 2 | False | 836.69 | True |
| 2 | 2 | False | 798.72 | True |
| 4 | 2 | False | 770.58 | True |
| 4 | 2 | True | 373.11 | True |

Primary forecast uses four bounded parallel readers and two host-prefetched batches. Device-prefetch results are measured separately below. No explicit pinned-memory allocation is claimed: TensorFlow controls transfers, and `prefetch_to_device` is the tested overlap mechanism. NumPy float32 division is preserved inside the pure cache-read callback to avoid graph rewrite to rounded reciprocal multiplication. Seeded sample/epoch stateless flips and deterministic order match the existing trainer.

## Training and utilization

| Stage / pipeline | measured s / 300 | training img/s | mean GPU % | mean CPU system % | process CPU % (100=one core) | peak GPU MiB | GPU <10% fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal_conv_1_11_cached | 13.82 | 21.71 | 60.5 | 5.0 | 92.1 | 8394 | 0.000 |
| normal_conv_1_14_cached | 10.72 | 27.98 | 58.8 | 4.9 | 89.1 | 8394 | 0.053 |
| normal_conv_1_17_cached | 7.99 | 37.52 | 66.2 | 4.1 | 77.5 | 4298 | 0.000 |
| normal_conv_1_17_device_prefetch | 7.82 | 38.38 | 70.3 | 4.1 | 77.0 | 8436 | 0.000 |
| normal_conv_1_17_online | 75.13 | 3.99 | 10.2 | 70.3 | 1539.8 | 4298 | 0.737 |
| normal_conv_1_17_pure | 7.58 | 39.56 | 71.1 | 4.2 | 75.6 | 8436 | 0.000 |
| normal_conv_1_1_cached | 21.71 | 13.82 | 59.1 | 4.5 | 86.9 | 8394 | 0.000 |
| normal_conv_1_1_device_prefetch | 23.19 | 12.94 | 66.8 | 4.9 | 92.0 | 8436 | 0.000 |
| normal_conv_1_1_online | 81.70 | 3.67 | 16.6 | 64.2 | 1343.9 | 8436 | 0.659 |
| normal_conv_1_1_pure | 21.77 | 13.78 | 61.4 | 9.1 | 86.3 | 8436 | 0.000 |
| normal_conv_1_3_cached | 19.51 | 15.37 | 63.3 | 4.6 | 87.3 | 8394 | 0.000 |
| normal_conv_1_5_cached | 17.56 | 17.08 | 63.0 | 4.7 | 90.1 | 8394 | 0.000 |
| normal_conv_1_7_cached | 16.21 | 18.50 | 60.0 | 4.8 | 91.8 | 8394 | 0.000 |
| normal_conv_1_9_cached | 15.30 | 19.60 | 58.9 | 4.9 | 92.7 | 8394 | 0.000 |

Utilization is sampled at approximately 0.5-second intervals; raw traces are retained. Low utilization is an idle proxy, not an exact stall percentage. Pure mode preloads float32 tensors onto GPU; online mode repeats strict JPEG preprocessing. Online/pure use synchronous execution, matching Stage 2D; cached/device-prefetch permit asynchronous dispatch with materialization at each completed microbatch/update. Therefore the speedup measures the safe optimization bundle, not an isolated cache-only causal effect.

### GPU computation components

| Stage | Forward s | Loss s | Backward s | Accumulation s | Mean Adam update s |
|---|---:|---:|---:|---:|---:|
| normal_conv_1_17 | 0.212835 | 0.004954 | 0.040288 | 0.019647 | 0.054052 |
| normal_conv_1_1 | 0.208184 | 0.005933 | 0.367195 | 0.041811 | 0.145831 |

Scope (normal_conv_1_17): Compiled graph with control-dependent CPU perf_counter markers; marker inputs materialize completed GPU values; backward includes gradient norm; diagnostic overhead differs from fused training

Scope (normal_conv_1_1): Compiled graph with control-dependent CPU perf_counter markers; marker inputs materialize completed GPU values; backward includes gradient norm; diagnostic overhead differs from fused training

First four columns total three diagnostic microbatches (12 images), after warmup. Adam is per logical 100-image update from fused training. Compiled phase diagnostics have materialized CPU timing markers and no allocated Adam slots. They include instrumentation/synchronization overhead and must not be added to predict ordinary fused training speed. An earlier eager first-stage probe is retained separately in the run artifacts.

## Separate validation inference

| Pipeline | images/s | mean GPU % | peak GPU MiB |
|---|---:|---:|---:|
| cached_compiled | 32.75 | 46.4 | 8445 |
| cached_eager | 5.38 | 25.8 | 8445 |
| online_compiled | 4.21 | 12.8 | 8445 |

Same 100 training images, three repeats, unchanged initial checkpoint, no augmentation/updates. Eager vs compiled and online vs cache predictions must satisfy predeclared forward tolerances and exact class agreement. Internal-validation images were not read; these are timing surrogates, not validation accuracy results.

## Revised runtime forecast

| Epoch(s) | Unfreeze boundary | Training img/s | Validation img/s | Training h/epoch | Validation h/epoch | Total h/epoch |
|---|---|---:|---:|---:|---:|---:|
| 1-7 | normal_conv_1_17 | 37.52 | 35.15 | 0.864 | 0.230 | 1.094 |
| 8 | normal_conv_1_14 | 27.98 | 31.24 | 1.158 | 0.259 | 1.417 |
| 9 | normal_conv_1_11 | 21.71 | 30.05 | 1.493 | 0.270 | 1.762 |
| 10 | normal_conv_1_9 | 19.60 | 30.93 | 1.653 | 0.262 | 1.915 |
| 11 | normal_conv_1_7 | 18.50 | 31.12 | 1.751 | 0.260 | 2.012 |
| 12 | normal_conv_1_5 | 17.08 | 33.53 | 1.897 | 0.242 | 2.139 |
| 13 | normal_conv_1_3 | 15.37 | 32.71 | 2.108 | 0.248 | 2.355 |
| 14 | normal_conv_1_1 | 13.82 | 32.27 | 2.345 | 0.251 | 2.596 |

Stage 2D serial estimate: **166.7-177.0 hours**. Stage 2E: one-time cache **8.87 h**; training **18.45 h**; validation **3.40 h**; final checkpoint **98.87 s**. Fourteen epochs including validation/checkpoint: **21.97 h**. Complete expected experiment including cache: **30.84 h**; conditional conservative planning range **30.84-40.88 h**.

Estimated model-loading/compilation allowance included in totals: 312.8 s.

- Steady state extrapolation from 300 measured training samples per stage after 100 warmup samples.
- Validation extrapolated from training images; no internal-validation or VAL2 pixels accessed.
- One final portable checkpoint, matching the existing production script; checkpoint timing excludes checksum verification.
- Upper planning bound: 30% cache/compute slowdown, 20 MiB/s cold shuffled input floor, double checkpoint/startup cost and one full-cache verification read at that assumed rate.
- Range is conditional, not a statistical confidence interval or guaranteed thermal/disk bound.
- Model load measured separately; startup estimate adds excess first-logical-batch warmup for each stage and compiled-validation warmup per stage.
- Full-cache post-generation verification and metadata startup costs remain unmeasured; allow operational slack beyond this conditional estimate.

## CPU/GPU numerical parity

Acceptance criteria were saved in `protocol.json` before measurements. No tolerance was relaxed after observing results.

```json
{
  "class_agreement": 1.0,
  "forward_logits_atol": 0.0001,
  "forward_logits_rtol": 0.0001,
  "forward_loss_atol": 0.0001,
  "forward_probability_atol": 1e-05,
  "forward_probability_rtol": 0.0001,
  "gradient_max_atol": 0.0001,
  "gradient_max_rtol": 0.001,
  "gradient_mean_atol": 1e-06,
  "gradient_mean_rtol": 0.0001,
  "gradient_relative_l2": 0.001,
  "post_loss_atol": 0.001,
  "post_probability_atol": 0.0001,
  "post_probability_rtol": 0.001,
  "relative_denominator_floor": 1e-08,
  "weight_max_atol_per_update": 5e-05,
  "weight_mean_atol": 1e-06
}
```

Float32 cross-device kernel/reduction differences are allowed, never bitwise identity. Forward tolerances are much smaller than the 0.5 decision threshold; class agreement is separately mandatory. Gradients use per-tensor absolute, relative-scale and relative-L2 checks; near-zero tensors use absolute floors. Adam near-zero gradients can amplify small reduction differences, so a per-update 5e-5 maximum weight bound is paired with a 1e-6 mean bound and post-update probability/loss checks. These are diagnostic acceptance criteria, not proof of equivalent 14-epoch trajectories. Declared before any CPU/GPU observations.

Shared checkpoint SHA-256: `079bc05f93030d6f5409dda662fa92c9c9d7c7a05c6d40d7186c484355e96e79`. Identical 100 cached images/labels, physical batch 4, fresh identical Adam, no augmentation. The CPU and GPU run in the same WSL package environment. Forward/post-update prediction probes use 16 of those images; training gradients use all 100.

Forward PASS: True; probability statistics: `{"elements": 32, "max_absolute": 1.1920928955078125e-06, "max_relative": 4.2502541120677256e-06, "mean_absolute": 3.7800054997205734e-07, "reference_l2": 3.0059385995871803, "reference_max_absolute": 0.9616597294807434, "reference_mean_absolute": 0.49999999755527824, "relative_l2": 9.91213419598418e-07, "rms": 5.267108797556624e-07}`; logits: `{"elements": 32, "max_absolute": 3.7848949432373047e-06, "max_relative": 2.966318042547745e-05, "mean_absolute": 1.2444797903299332e-06, "reference_l2": 4.733167663907244, "reference_max_absolute": 2.228280544281006, "reference_mean_absolute": 0.6589933880604804, "relative_l2": 1.9088200296344996e-06, "rms": 1.5971359419937714e-06}`; loss difference 0; class agreement 1.000000.

Gradients: 39/47 tensors passed. Maximum absolute difference across tensors: 0.00051503628. Per-tensor max/mean/relative/L2 statistics and shapes are retained in JSON; buffer index follows model trainable-variable order.

| Update | CPU pre-loss | GPU pre-loss | Weight max abs | Weight mean abs | Post probability max abs | Post class agreement | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.84622195 | 0.84622214 | 1.8605962e-05 | 7.1424452e-11 | 1.0728836e-06 | 1.000000 | True |
| 2 | 0.47455196 | 0.47455141 | 2.9503368e-05 | 1.4477159e-10 | 1.5497208e-06 | 1.000000 | True |
| 3 | 0.05442787 | 0.05442782 | 3.1745061e-05 | 2.0643346e-10 | 4.1723251e-07 | 1.000000 | True |

### Gradient failures (unchanged acceptance limits)

| Variable | Max absolute difference | Mean absolute difference | Relative L2 | Failed criterion |
|---|---:|---:|---:|---|
| normal_A_block_17/normal_conv_1_17/kernel | 0.0001572998 | 1.0180229e-07 | 0.0012710817 | relative L2 bound |
| normal_A_block_17/block_2/separable_conv_block_normal_right2_17/separable_conv_1_normal_right2_17/depthwise_kernel | 2.7127564e-05 | 1.83779e-07 | 0.0010581332 | relative L2 bound |
| normal_A_block_17/block_5/separable_conv_block_normal_left5_17/separable_conv_1_normal_left5_17/depthwise_kernel | 7.0432201e-05 | 6.1531807e-07 | 0.0012606688 | relative L2 bound |
| normal_A_block_17/block_5/separable_conv_block_normal_left5_17/separable_conv_1_normal_left5_17/pointwise_kernel | 5.9994869e-05 | 1.5496316e-07 | 0.0012407464 | relative L2 bound |
| normal_A_block_17/block_2/separable_conv_block_normal_right2_17/separable_conv_2_normal_right2_17/depthwise_kernel | 1.0260381e-05 | 2.3868893e-07 | 0.0010136078 | relative L2 bound |
| normal_A_block_17/block_5/separable_conv_block_normal_left5_17/separable_conv_2_normal_left5_17/depthwise_kernel | 4.0331579e-05 | 6.7652601e-07 | 0.001157817 | relative L2 bound |
| normal_A_block_17/block_5/separable_conv_block_normal_left5_17/separable_conv_2_normal_left5_17/pointwise_kernel | 3.5154168e-05 | 1.5831829e-07 | 0.0013957805 | relative L2 bound |
| normal_A_block_18/normal_conv_1_18/kernel | 0.00051503628 | 2.8576105e-08 | 0.00084485611 | maximum absolute / tensor-scale bound |

Names are mapped from saved model-weight ordering after the first unfreeze boundary, excluding frozen BatchNorm variables; all 47 counts/shapes were verified. These are real failures of the registered criteria even though predictions and post-update weights pass.

Next investigation: localize failing sample/layer gradients, verify repeatability within each device, and inspect near-zero activation boundaries and CPU/CUDA reduction differences. These are hypotheses, not established causes. Any alternative CPU backend or acceptance policy requires a separately identified follow-up; do not retroactively relax these limits.

CPU logical-update seconds: [36.365989746991545, 27.01839510400896, 26.40633734599396]. Predeclared practicality rule: if the first CPU update exceeds 900 seconds, omit subsequent updates. Overall numerical parity: **FAIL**. Individual failure details remain in JSON; raw logits/probabilities/losses are in `parity_cpu/result.json` and `parity_gpu/result.json` under the run directory. Short-run agreement never proves identical 14-epoch trajectories.

## Recommendations and remaining gates

- Full-cache generation remains unapproved and unperformed.
- Cache loader is a bounded profiling implementation, not yet integrated into the production trainer.
- Full-cohort cache identity, complete coverage and cold shuffled I/O must be verified before launch.
- CPU/GPU parity failed predeclared acceptance criteria; investigate before production.

If approved later, generate and verify separate full train/internal-validation caches; integrate this verified loader into the separate reimplementation trainer; retain original shuffling, stateless flips, full sample coverage, final partial logical-batch handling and staged Adam resets. Measure cold shuffled reads and sustained laptop thermals before scheduling the full run. Keep VAL2 sealed.

## Limitations

- 1000-image format benchmark fits OS cache; warm mmap rates are not cold full-cache disk performance.
- Execution spans October 1-2 after a profiler failure/resume; laptop thermal/power variation is not controlled between jobs.
- Training timings use the first 400 interleaved subset images (200 per class); the complete 1000-image profiling subset follows the training class ratio.
- Validation timings include construction/iteration of small 100-image datasets; fixed setup overhead makes per-image extrapolation conservative.
- GPU utilization <10% is a sampled idle proxy, not measured kernel stall time; nvidia-smi includes other desktop processes.
- CPU system percent spans all logical CPUs; process percent uses one-core=100%.
- GPU memory from nvidia-smi includes allocator reservations and other processes, not only live model tensors.
- Diagnostic forward/backward phase timers use a different execution graph from fused training; do not sum them to predict epochs.
- CPU/GPU parity uses Linux CPU and GPU in the same WSL environment, not a Windows/WSL environment-equivalence claim.
- Short-update parity does not establish equivalent long training trajectories.
- The benchmark uses deterministic training subsets; it does not measure predictive generalization.

## Reproducibility and checks

Profiling code: `modern_pca/preprocessing_cache.py`, `modern_pca/profile_reimplementation.py`, `modern_pca/profile_validation.py`, `modern_pca/numerical_parity.py`. Report builder: `tools/report_stage2e.py`. Tests: `tests/test_stage2e.py`.

The first preparation attempt completed cache parity, then stopped on an ndarray/tensor return-type handling error in the profiling check. The check was corrected and resumed from the verified cache. Original protocol and logs were retained; `resume_implementation.json` records the new profiling-source fingerprint. Scientific preprocessing and predeclared tolerances were unchanged.

A later phase-by-phase diagnostic exhausted GPU memory when its additional gradient graph coexisted with the trained model/Adam slots. Cached training at all eight stages had already succeeded. A fresh eager diagnostic passed at the first boundary but exhausted memory at the deepest boundary. Final component diagnostics use compiled graph memory planning with control-dependent, materialized CPU perf_counter markers. Failed logs are preserved; resume records bind profiling-source versions. Fused training configuration was unchanged.

The initial CPU logit-only probe used an eager batch of 16 and exhausted memory before any optimizer update. Final CPU/GPU logits are computed in compiled batches of four, matching the probability/training probe physical batch. Initialization, 100-image logical updates and acceptance tolerances were unchanged.

Protected artifacts are rehashed by the report builder. Test and Git check evidence is recorded separately in `reports/stage2e_checks.json`.

TensorFlow references: [input pipeline performance](https://www.tensorflow.org/guide/data_performance), [performance analysis](https://www.tensorflow.org/guide/data_performance_analysis), [prefetch_to_device](https://www.tensorflow.org/api_docs/python/tf/data/experimental/prefetch_to_device).
