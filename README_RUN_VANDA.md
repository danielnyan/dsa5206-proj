# NASNetLarge from-scratch ablation: Vanda runbook

**Scope:** binary tumour-versus-benign classification using the *same cached VAL1 train/internal-validation split* as the team's ImageNet-initialized native binary checkpoint, and an isolated post-training VAL2 evaluation. **Not** PANDA Gleason GP3/4/5. The `--initialization scratch` option means Keras' random initializers (`weights=None`), not uninitialized tensors.

## What is controlled and what changes

- Input: authenticated preprocessed Stage-2G VAL1 float32/255 cache, exactly 116,655 training + 29,164 internal-validation patches.
- Architecture: NASNetLarge (no ImageNet weights when `scratch`), Flatten → Dense(256, ReLU) → Dense(2, softmax), **209,812,820** parameters.
- **Every backbone layer is trainable from epoch 1**, including BatchNorm. The backbone executes with `training=True` on training batches. This contrasts with the old staged-unfreezing / inference-mode-BN native binary run.
- Optimizer: Adam, default learning rate `1e-4`, sample-average cross-entropy, effective batch 100 (physical batch 4, 25 gradient-accumulation microbatches, last group of epoch flushed). Fixed rate for 14 epochs; no staged changes. Seed 42. Stateless image flips identical to the native-cache training pipeline.
- Measurements: epoch-by-epoch training loss/accuracy, internal VAL1 validation loss/accuracy plus the native binary evaluation metrics. The final VAL2 cross-entropy + metrics are recorded in an **explicit separate final-test job**, to prevent use of test data for training.
- Resumption: checkpoint saved after each full epoch with all trainable/nontrainable weights and Adam state. A resumed run with modified code, cache or settings is rejected.
- Optional matched-initialization control: `SCRATCH_INIT=imagenet` uses ImageNet initialization **but the same all-unfrozen schedule** as scratch. This is a stronger comparison than using only the earlier staged-fine-tuning native checkpoint. Two runs are costly; start with the scratch epoch-1 diagnostic if time-limited.

## Install new files (from ZIP) in existing cloned repository

```bash
cd "$REPO_DIR"
cp /path/to/unzipped/hpc/nasnet_binary_initialization_ablation.py hpc/
cp /path/to/unzipped/hpc/pbs/nasnet_binary_initialization_ablation.pbs hpc/pbs/
```

The experiment depends on existing repository modules (`modern_pca/models.py`, `modern_pca/preprocessing_cache.py`, `modern_pca/vanda_cache_consumer.py`, `modern_pca/reimplementation.py`, `modern_pca/evaluate_paper.py`) and the **verified Stage-2G VAL1 cache**. It does not build/reprocess the 54-GB cache. Do not point `VAL1_CACHE` at PANDA `PREPARED_DIR` or an arbitrary `.npy` folder. Run under the environment that can load the earlier native trainer and cache.

## Submit from the login node

```bash
export PIPELINE_CONFIG=/scratch/USER/site.env
source "$PIPELINE_CONFIG"
export VAL1_CACHE=/absolute/path/to/verified/VAL1/Stage2G/cache
export SCRATCH_OUTPUT="$RUN_ROOT/nasnet-binary-scratch-v1"
export SCRATCH_MODE=train
export SCRATCH_INIT=scratch
export SCRATCH_LR=0.0001
export SCRATCH_MAX_EPOCHS=1

qsub -v "PIPELINE_CONFIG=$PIPELINE_CONFIG,VAL1_CACHE=$VAL1_CACHE,SCRATCH_OUTPUT=$SCRATCH_OUTPUT,SCRATCH_MODE=$SCRATCH_MODE,SCRATCH_INIT=$SCRATCH_INIT,SCRATCH_LR=$SCRATCH_LR,SCRATCH_MAX_EPOCHS=$SCRATCH_MAX_EPOCHS" \
 hpc/pbs/nasnet_binary_initialization_ablation.pbs
```

**Important:** PBS `qsub -v` propagates explicitly named variables; `export` in the interactive login shell does **not** automatically propagate to PBS. Variable values here must contain no commas. If your site requires a PBS project/account or `-q auto`, add the site-required `#PBS` lines locally to the launcher. Verify Vanda's queue walltime limits. For a batch-4 OOM, start a **fresh output directory** with `SCRATCH_BATCH=2`, exported in `qsub -v`, before retrying. Never silently change the batch size mid-resume.

When the first epoch finishes inspect:

```bash
cat "$SCRATCH_OUTPUT/progress.json" | head -40
cat "$SCRATCH_OUTPUT/loss_curves.csv"
```

Then resubmit with the same output, signature and code, changing **only** `SCRATCH_MAX_EPOCHS=14`. The script resumes from the most recent committed epoch. It stores `epoch_01`, the last two checkpoints, and an epoch-14 `.keras` export; sufficient scratch disk capacity is required. If the scheduler interrupts an epoch, that partial epoch is repeated upon resubmission.

Use the same `qsub` command with `SCRATCH_MAX_EPOCHS=14`. Re-submit as required until `final.json` reports `complete: true`.

## Final held-out VAL2 cross-entropy

This phase is **not** started automatically. Requires extracted Zenodo VAL2 images in the layout that `modern_pca.reimplementation.image_path` expects, and the frozen `manifests/VAL2_manifest.csv` in the repository.

```bash
export SCRATCH_MODE=test
export VAL2_EXTRACTED=/absolute/path/to/extracted/Zenodo/data
export SCRATCH_OUTPUT="$RUN_ROOT/nasnet-binary-scratch-v1"
export SCRATCH_TEST_OUTPUT="$RUN_ROOT/nasnet-binary-scratch-v1-val2-final"

qsub -v "PIPELINE_CONFIG=$PIPELINE_CONFIG,SCRATCH_MODE=$SCRATCH_MODE,VAL2_EXTRACTED=$VAL2_EXTRACTED,SCRATCH_OUTPUT=$SCRATCH_OUTPUT,SCRATCH_TEST_OUTPUT=$SCRATCH_TEST_OUTPUT" \
 hpc/pbs/nasnet_binary_initialization_ablation.pbs
```

Results: `$SCRATCH_TEST_OUTPUT/test_metrics.json` (contains `test_cross_entropy`), `predictions.csv`. VAL2 is evaluated **only after** 14 completed training epochs and should not be used for learning-rate/model selection. Note that the wider project previously evaluated VAL2 with other models, so be careful with claims of complete researcher-blindness.

## Comparison matrix to report

| Run | Initial weights | Backbone trainable at epoch 1? | LR/schedule | BN behaviour | First-epoch test? | Final test? |
|---|---|---|---|---|---|---|
| Prior native 14-epoch binary model | ImageNet | No, progressively unfrozen | 1e-5 then 1e-6 | Frozen/inference | historical history (if retained) | prior VAL2 metrics; **test cross-entropy not supplied** |
| New scratch pilot / 14-epoch run | Random | **Yes, all** | fixed 1e-4 (configurable) | trainable / training mode | this experiment | only after epoch 14 |
| Optional matched ImageNet control | ImageNet | **Yes, all** | same as scratch | same as scratch | future | future |
| Colleague's ESCNN | Report architecture, training/splits and exact parameter count | n/a | colleague metadata | colleague metadata | pending | pending |

**Interpretation warning:** scratch versus the *old* pretrained native run conflates initial weights with unfreezing schedule, LR and BN handling. Only scratch versus optional matched ImageNet all-unfrozen run isolates initialization reasonably well. One epoch can show **learning from random initialization**, not that ImageNet pretraining confers no benefit at 14 epochs, on other seeds, or on external data.

## Status and testing

- Source file syntax validated by `py_compile` in the authoring environment.
- Integration against a real TensorFlow GPU, Vanda modules, 54-GB cache and VAL2 extracted data **has not been executed here**. Do a one-epoch smoke/pilot before committing to 14 epochs.
- Actual training throughput and walltime are unknown: full NASNetLarge backward passes from epoch 1 may be substantially more expensive than the old partially-unfrozen schedule. One elapsed epoch is necessary for an ETA.
