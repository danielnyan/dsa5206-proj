# Stage 2G full VAL1 cache

Generated 2026-10-02T14:18:49.327252+00:00

Full cache: **PASS**. Trainer/cache integration: **PASS**. Production: **GO; recommendation only, production not started**.

Cache: `/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/GitHub/dsa5206-proj/runs/stage2g_full_cache/cache`

| Item | Verified value |
|---|---:|
| Total samples | 145,819 |
| Training | 116,655 |
| Internal validation | 29,164 |
| NPY shards | 570 |
| NPY bytes | 53,588,555,460 |
| Metadata bytes | 149,393,342 |
| Total cache GB (decimal) | 54.114269 |
| Active cache GB (excluding preserved partials) | 53.737949 |
| Preserved interrupted files / bytes | 4 / 376,320,512 |
| Generation hours | 2.8093 |
| Generation images/s | 14.418 |
| Failures / automatic retries | 0 / 4 |

## Provenance and integrity

Source manifest, both frozen splits, stain reference, preprocessing fingerprint and dependency pins were verified before generation. Full metadata is in the JSON report and cache context. Items retain sample ID, label, split, original relative path and JPEG SHA-256, tensor SHA-256, shard/index, uint8 shape and preprocessing status.

Each cache tensor is 350×350×3 uint8 after the unchanged PIL RGB → Pillow LANCZOS → legacy brightness → StainTools Macenko path. Float32 conversion, /255 and stateless flips occur after loading. Validation is never augmented.

All 145,819 tensor checksums and original VAL1 source JPEG checksums were reverified. Source size/mtime were also checked against generation. Full mapping, counts, shape/dtype, orphan/incomplete-shard and uniqueness checks passed.

Representative audit: **PASS**, 590 items covering both splits/classes, native dimension strata and every shard. Deterministic random audit: **PASS**, 1,000 items. Both uint8 tensors and the strict online float32 preprocessing output matched exactly: maximum difference 0; differing elements 0.

Committed shards are checksum-verified and retained on resume. Uncommitted shard remnants are preserved in `interrupted/` before rebuilding. The current implementation assumes one cache-generation writer at a time.

## Trainer integration and bounded dry run

The actual trainer input path traversed all 145,819 items with exact order/label/split checks. Physical batch 4, effective batch 100, float32, TF32 disabled, frozen BN, Adam settings and the 14-epoch schedule remain unchanged.

Dry run: 100 training samples, exactly one optimizer update, and 32 internal-validation samples. Forward/backward/accumulation and outputs were finite. Frozen weights stayed exact. Model and Adam state were written, deliberately perturbed, restored and hash-checked; predictions matched exactly after restoration. The resume cursor returned the next four expected samples without another optimizer update.

A disposable full NASNetLarge `.keras` export was also written and reloaded. All weight hashes and the probe prediction matched exactly; the diagnostic artifact is not a production checkpoint.

The cache path is optional in `modern_pca/train_reimplementation.py`; future production requires explicit `--train-production --cache ... --batch-size 4`. The separate `modern_pca.cached_training` CLI performs only the disposable one-update dry run. Production rejects its diagnostic checkpoint.

## Performance and forecast

| Measurement | Stage 2E estimate/measurement | Stage 2G actual/revised |
|---|---:|---:|
| NPY GB | 53.588555 | 53.588555 |
| Metadata MB | 167 (approximate) | 149.393 |
| Generation hours | 8.87 (serial estimate) | 2.8093 (4 workers) |
| Loader images/s | 770.6 (bounded subset) | 130.835 (full cache) |
| Training-only hours | 21.969 | 22.087 (forecast) |

Sum of perf_counter worker-pool session wall times; excludes preflight, committed-shard resume validation and final catalog/verification scans; downtime excluded.

Interrupted sessions: 1. Timing is a lower bound: True. An abruptly interrupted session retains its last persisted monotonic measurement, excluding its unrecorded tail; in that case the reported generation throughput is correspondingly an upper bound, not an exact timing claim.

Worker CPU utilization summed across workers: 2127.56% of one core. Payload write rate averaged over generation: 5.299 MB/s; this includes compute time and is not a block-device benchmark.

Full-loader timing: full input iteration, scaling/flips, host materialization, finite/order/label checks; no model. Cache temperature: uncontrolled OS cache after integrity audit; no cold-cache claim. Reader startup took 9.464 seconds.

The forecast includes 14 epoch-resume checkpoints in addition to the final portable model. The conservative serial-input bound is 26.481 hours.

- Compute rates retain Stage 2E measurements; full production training was not benchmarked.
- Primary estimate assumes input prefetch overlaps compute; serial bound adds both and may double-count part of Stage 2E input cost.
- Checkpoint cost uses the first-stage dry-run write plus checksum verification for all 14 epoch checkpoints; serial bound allows 50% more for larger later-stage Adam state.
- Not a confidence interval; thermal variation, background IO and compile/checkpoint variation remain possible.
- Generation and integrity audit are one-time preparation and excluded from training-only hours.

## Scope and checks

VAL2 accessed: **NO**. Source images modified: **NO**. Strict evaluator modified: **NO**. Frozen split changed: **NO**. Production training started: **NO**. Commit/push: **NO**.

Stage 2F strict CPU/GPU parity remains **FAIL** under its original thresholds; its separate numerical GO assessment is unchanged.

Tests, source hashes, `git diff --check`, new-file whitespace checks and final Git status are recorded in `reports/stage2g_checks.json`.
