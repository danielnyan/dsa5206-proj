# Stage 2H epoch-12 OOM investigation

**Recommendation: A — physical batch 4 + `TF_GPU_ALLOCATOR=cuda_malloc_async`; effective batch 100 and validation batch 2.**

No production run was started. All optimizer updates belong to bounded disposable smoke tests. Existing production checkpoints/history and cache metadata stayed unchanged. GPU probes and tests recorded zero real VAL2 access attempts; no VAL2 images were opened or evaluated. A repository-wide git diff --check did inspect tracked VAL2 manifest files, contrary to the requested no-VAL2 scope. Their contents were not used in recovery decisions. No commit/push was performed.

## Root cause and recovery state

Epoch 11 is complete at `normal_conv_1_7`, Adam iteration 1167. Strict restoration verifies checkpoint-file, model-tensor and optimizer-tensor hashes. The next production epoch is 12, boundary `normal_conv_1_5`, LR 1e-6, with **206,213,042 trainable parameters** and a fresh Adam optimizer.

The failing operation `truediv_331` is the accumulator buffer for `paper_dense_256/kernel` divided by the actual logical sample count. Its `[487872,256]` float32 result requires **499,580,928 bytes = 476.44 MiB**. The existing list-based apply materializes averaged gradient tensors while model weights, accumulated sums and Adam slots remain resident. Adam slots are built lazily at the first apply.

The original run has no explicit epoch-start or per-flush cursor. Epoch 12 starts after checkpoint completion at 2026-10-07 01:37:31.036 Singapore time and fails at 01:38:18.192. No 1000-sample progress message appears: the failing flush corresponds to 100–1000 samples (updates1–10), because progress is logged only after a successful flush. The first flush is plausible but cannot be established exactly from these logs.

The user-provided allocator summary reports ~9.35GB in use against a ~9.968GB pool, a ~9.95GB high-water mark and no largest free block. The failed request is ~0.500GB. This supports fragmentation/allocator-history pressure combined with high live memory, rather than proving that the frozen epoch-12 model inherently exceeds capacity. The fresh-process baseline completed its bounded updates, so the original OOM did **not** reproduce there. No claim is made to have replicated hours of allocator history.

At normal transitions, the source deletes the accumulator/optimizer and calls `gc.collect()`. Cross-stage resume deletes the old Adam after strict restoration. Weak-reference observations and GPU memory snapshots are recorded per test; no speculative production lifetime/allocator code was changed.

## Candidate results

| Candidate | Epoch12 | Epoch13 | Epoch14 | Max GPU peak (GB) | Steady images/s (12 / 13 / 14) | Numerical equivalence | Protocol deviation | Recommendation |
|---|---|---|---|---:|---|---|---|---|
| A | PASS | PASS | PASS | 5.857 | 16.59 / 14.49 / 13.39 | PASS, quantified below | Allocator only; training 4/effective100, validation2 unchanged | Recommended |
| B | PASS | PASS | PASS | 5.812 | 10.97 / 9.63 / 8.94 | PASS, quantified below | Physical training batch2 instead of frozen4; effective100 unchanged | Not selected |
| C | PASS | PASS | PASS | 6.287 | 12.56 / 10.75 / 10.89 | PASS, quantified below | Experimental serialized buffer averaging; Adam mathematics unchanged | Not selected |

Every independent stage smoke restores epoch 11 and uses 300 deterministic epoch-specific VAL1 training samples: 75 microbatches at physical4, or 150 at physical2, and exactly three 100-sample updates. Epoch13/14 independent probes use epoch11 weights as disposable initial test state, not claims of completed production epochs. Each successful stage also checks inference on eight internal-validation samples at batch2.

Same-process transition checks additionally carry disposable updated weights through stages12→13→14 with fresh Adam at each stage, without resetting the allocator or clearing the session. Each stage uses **355 samples (100,100,100,55)**, exercising the final short microbatch/retrace, inference and full disposable checkpoint writing before the next stage. The final disposable model/Adam checkpoint is strictly restored again:
- `baseline_transitions_12_14`: **OOM**; transition details and memory snapshots in JSON.
  - Stage 12: training memory current/peak bytes `{'current': 3236439808, 'peak': 5833386496}`; object lifetimes `not instrumented`.
  - Stage 13: training memory current/peak bytes `{'current': 5490758912, 'peak': 8310701824}`; object lifetimes `not instrumented`.
  - Stage 14: training memory current/peak bytes `None`; object lifetimes `not instrumented`.
- `A_transitions_12_14`: **PASS**; transition details and memory snapshots in JSON.
  - Stage 12: training memory current/peak bytes `{'current': 3021299880, 'peak': 5347120128}`; object lifetimes `not instrumented`.
  - Stage 13: training memory current/peak bytes `{'current': 5213695692, 'peak': 7784088996}`; object lifetimes `{'after_collection': {'current': 3021299580, 'peak': 5548108828}, 'before_collection': {'current': 3021299884, 'peak': 5548108828}, 'old_accumulator_live': False, 'old_optimizer_live': False}`.
  - Stage 14: training memory current/peak bytes `{'current': 7416440300, 'peak': 10231594812}`; object lifetimes `{'after_collection': {'current': 5213695388, 'peak': 7784088996}, 'before_collection': {'current': 5213695692, 'peak': 7784088996}, 'old_accumulator_live': False, 'old_optimizer_live': False}`.

Corrected baseline transition failure: phase `epoch14_microbatch`, current/peak bytes `{'current': 7941652480, 'peak': 9942699520}`. Stage 14 completed its first optimizer update, then failed in the next microbatch at `AssignAddVariableOp_419`, accumulating the `[487872,256]` dense tensor. This reproduces transition-related memory pressure, but is a different operation/stage from the original epoch-12 division OOM. It does not establish fragmentation as the sole cause.

In the async transition probe, old accumulator and optimizer weak references were both dead after collection, yet live GPU allocation decreased by only 304 bytes at each transition. Python object collection therefore does not imply release of all TensorFlow runtime resources. The remaining ownership was not isolated; attributing the growth solely to fragmentation or a specific leaked Python optimizer would exceed the evidence. No production resource-lifetime code was changed.

## Numerical checks

The small deterministic frozen-layer/BN fixture uses 355 samples: effective groups **100,100,100,55**, four updates, no dropped samples, finite gradients/losses, frozen-weight hashes and checkpoint save/perturb/restore verification. Predeclared tolerances: gradients atol1e-6/rtol1e-5; weights/predictions atol1e-7/rtol1e-5. Tolerances were not loosened.

| Candidate vs baseline fixture | Max absolute gradient difference | Max absolute weight difference | Max absolute prediction difference | Class agreement |
|---|---:|---:|---:|---|
| A | 0 | 0 | 0 | True |
| B | 1.11758709e-08 | 2.27373675e-13 | 0 | True |
| C | 0 | 0 | 0 | True |

Full NASNet epoch12 baseline vs async: `{"same_sample_ids": true, "identical_weight_hashes": true, "max_absolute_validation_probability_difference": 0.0, "max_absolute_loss_difference": 0.0}`.
Same-process transition checkpoint model/Adam hash comparisons: `{"12": {"model_hashes_identical": true, "optimizer_hashes_identical": true}, "13": {"model_hashes_identical": true, "optimizer_hashes_identical": true}}`.
Other epoch12 full-model comparisons: `{"B": {"identical_weight_hashes": false, "max_absolute_validation_probability_difference": 1.1920928955078125e-07, "max_absolute_loss_difference": 3.5387975855227793e-08}, "C": {"identical_weight_hashes": true, "max_absolute_validation_probability_difference": 0.0, "max_absolute_loss_difference": 0.0}}`. Full weight hashes measure bitwise agreement; magnitude of B weight differences is measured on the deterministic fixture, not inferred from hashes.
Exact trainable variable names/shapes, ordered sample IDs and zero initial new-stage Adam iterations are checked across A/B/C at every stage.

## Tests and working tree

Focused: 64 passed in 11.86s. Full: 254 passed in 58.41s. Both run CPU-only after GPU probes, with real VAL2 paths blocked (synthetic pytest fixtures allowed).
Git diff --check: PASS, including no-index whitespace checks for the new Python files. Existing tracked validation-batch changes were preserved. This investigation adds diagnostic tools, tests and reports, and extends the existing resume orchestration test to cover epoch11 -> epoch12 with history1..12 exactly once. No production source was changed in this investigation.

```text
 M modern_pca/cached_training.py
 M modern_pca/train_reimplementation.py
 M tests/test_stage2h_readiness.py
?? reports/stage2e_report_generation.log
?? reports/stage2e_tests.log
?? reports/stage2f_diagnostic_tests.log
?? reports/stage2f_report_generation.log
?? reports/stage2f_tests.log
?? reports/stage2g_cache_generation.log
?? reports/stage2g_focused_tests.log
?? reports/stage2g_tests.log
?? reports/stage2h_epoch12_focused_tests.log
?? reports/stage2h_epoch12_full_tests.log
?? reports/stage2h_epoch12_git_status.txt
?? reports/stage2h_epoch12_oom_investigation.json
?? reports/stage2h_epoch12_oom_investigation.md
?? reports/stage2h_focused_tests.log
?? reports/stage2h_tests.log
?? reports/stage2h_validation_batch2_verification.json
?? reports/stage2h_validation_batch2_verification.log
?? reports/stage2h_validation_batch_verification.json
?? reports/stage2h_validation_batch_verification.log
?? reports/stage2h_validation_focused_tests.log
?? reports/stage2h_validation_full_tests.log
?? tests/test_stage2h_oom_candidates.py
?? tools/report_stage2h_epoch12.py
?? tools/stage2h_epoch12_probe.py
?? tools/verify_stage2h_validation.py

```

## Warnings and limitations

After the focused/full tests completed successfully, pytest atexit temporary-directory cleanup emitted two audit-guard exceptions on relative VAL2-named entries. The conservative guard blocked those opens; no blocked open proceeded. The suites had already reported 64 and 254 passing tests, zero real VAL2 attempts and exit0. This cleanup warning is retained separately from the test outcomes.

Console logs retain TensorFlow startup/oneDNN notices and any cuDNN heuristic fallback warnings. A fallback warning occurred in the batch2 epoch13 run; the test still completed with finite losses/gradients. No warnings were suppressed and no numerical tolerances were relaxed. Candidate C is a tested hypothesis, not an assumed improvement: its serialized averaging may have a higher actual peak/compile cost despite reducing an explicit simultaneous gradient list.

An initial baseline transition diagnostic retained an inspection-only ConcreteFunction reference across stages. That artificial resource retention was identified, the probe interrupted and preserved, and the reference removed. Only the corrected `baseline_transitions_12_14_clean` results are used here; the async transition probe also uses the corrected harness.

## Reproduction commands and environment

Environment/build/GPU details, allocator environment, exact worker argument lists, sample IDs, checkpoint/history hashes, loss per update, compile-inclusive and steady timings, current/peak memory and error tracebacks are in the JSON report. Console logs retain allocator startup messages and warnings.

From the repository in WSL, each worker was invoked with the verified environment Python and:

```text
python -B -u -m tools.stage2h_epoch12_probe --candidate {baseline,A,B,C} --epoch {12,13,14} --updates 3 --output <new isolated directory>
# A only: TF_GPU_ALLOCATOR=cuda_malloc_async
# Numerical fixtures: --fixture instead of an epoch test
# Same-process transition tests: --epoch 12 --chain
```

Outputs are under `runs/stage2h_epoch12_oom_smoke_20261007/`. Candidate C exists only in the diagnostic harness and is not wired into production. Candidate B bypasses the production CLI only within the bounded diagnostic.

## Proposed production resume — NOT EXECUTED

The command resumes epoch12 from the latest complete epoch11 checkpoint. It uses a new output directory. No architecture, preprocessing, sample order, augmentation, Adam mathematics, LR, schedule, BN policy or batch semantics change.

```bash
cd '/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/GitHub/dsa5206-proj'
TF_GPU_ALLOCATOR=cuda_malloc_async \
  /home/soonh/miniconda3/envs/dsa5206-reimplementation/bin/python -B -u \
  -m modern_pca.train_reimplementation \
  --train-production \
  --manifest manifests/VAL1_manifest.csv \
  --split-output manifests/VAL1_internal_v1 \
  --cache runs/stage2g_full_cache/cache \
  --batch-size 4 \
  --validation-batch-size 2 \
  --resume-cache-checkpoint runs/stage2h_production_resume_epoch08_vbatch2/epoch_11_resume \
  --output runs/stage2h_production_resume_epoch11_cuda_malloc_async
```

Bounded feasibility is not a guarantee against an OOM after an entire long-running epoch. The recommendation is supported by all remaining stages and the same-process transition smoke, with no physical-batch protocol deviation.
