# HPC runbook

## 1. One-time setup

Run `cd "/scratch/USER"`, where USER is your student ID (starting with eXXXXXXX), 
lowercase, and clone this repository using 
`git clone -b gleason https://github.com/danielnyan/dsa5206-proj`. 
From the extracted repository, copy the example config outside it:

```bash
cp hpc/site.env.example /scratch/USER/site.env
export PIPELINE_CONFIG=/scratch/USER/site.env
```

Edit only `REPO_DIR`, `WORK_ROOT`, site module commands if needed, and the
Crowd/SICAP `SICAP_WSI_LABELS` path to replace USER with your student ID. 
Additionally, stage `wsi_labels.xlsx` by running the following while you are in 
`/scratch/USER`: 
```bash
mkdir raw/SICAPv2
cp dsa5206-proj/supplementary-data/wsi_labels.xlsx raw/SICAPv2/wsi_labels.xlsx
```

Other paths are derived automatically and
track-specific. If upgrading, refresh your old configuration from this example:
hardcoded old dataset/output paths would defeat automatic track selection.

Adjust queue/account/resource headers in `hpc/pbs/*.pbs` for your cluster.
The supplied headers use PBS Pro/OpenPBS. On an approved interactive CPU
allocation if your site prohibits installation on the login node:

```bash
bash hpc/bootstrap_env.sh
```

Both preparation routes download inside their CPU PBS job. Compute nodes need
outbound HTTPS; otherwise use an approved network-enabled CPU queue. Allow at
least approximately 100 GB shared scratch for the large track. No full data,
model training, extraction or normalization runs on the login node.

## 2. Prepare, then GPU smoke

Choose one track; there is no need to run both simultaneously:

```bash
bash hpc/submit_pipeline.sh crowd-sicap smoke
bash hpc/submit_pipeline.sh gleason2019 smoke
```

The helper queues CPU preparation followed by a dependent GPU smoke job. It
reuses an existing completed preparation; it never overwrites an incomplete
output. For preparation only, replace `smoke` with `prepare`.

Crowd/SICAP downloads released images/annotations plus ImageNet weights.
Gleason2019 downloads its training JPEGs and all six expert annotation archives,
cuts native 600x600 tiles, and prepares 350x350 images. The weights downloader's
`weights-only` mode avoids downloading the other track's data.

Check PBS logs and the smoke run under `WORK_ROOT/runs/TRACK/JOB-smoke`.
Smoke enables TensorFlow numerical checks and explicitly disables Keras JIT
compilation because XLA GPU compilation does not support their debug operations.
This applies to both tracks. Production retains Keras's default compile settings.
A successful smoke output has `run.json` with `complete: true`, two completed
stages in `checkpoint.json`, stage CSV metrics, and `final.keras`; completion is
written after saved-model reload predictions match. Batch-1 smoke success does
not establish that the default production batch of 100 fits GPU memory.
Gleason2019 `prepared.json` records class counts and training readiness; all
three GP classes must occur in the seed training cohort. Insufficient coverage
fails explicitly, without silently changing the split or eligibility policy.

## 3. Full initial training

After the corresponding smoke run passes:

```bash
bash hpc/submit_pipeline.sh crowd-sicap train
bash hpc/submit_pipeline.sh gleason2019 train
```

Preparation is reused. No production run is automatically started. Defaults
preserve physical batch 100 and the legacy 14-epoch schedule; batch 1 is used
only for smoke. If batch 100 does not fit, a smaller batch is a recorded
methodological deviation, not an automatic fallback.

## 4. Evaluate or mine

Set the completed production model directory from the **same track**:

```bash
export MODEL_DIR=/scratch/USER/dsa5206-work/runs/gleason2019/JOB-production
bash hpc/submit_pipeline.sh gleason2019 evaluate
bash hpc/submit_pipeline.sh gleason2019 mine
```

Use `crowd-sicap` and its own model path for that track. Evaluation checks
frozen preparation hashes. Mining accepts only a complete non-smoke pure-cohort
teacher, predicts only multipattern training sources, and exports strict
GP probability >0.95 selections separately from expert annotations. It does
not train on that expansion pool automatically.

To resume a production run from its last completed training stage:

```bash
qsub -v PIPELINE_CONFIG=$PIPELINE_CONFIG,DATASET=gleason2019,MODE=production,RESUME=1,MODEL_DIR=$MODEL_DIR hpc/pbs/train_gleason.pbs
```

## PBS files

| File | Allocation | Purpose |
|---|---|---|
| `prepare_gleason.pbs` | CPU | Automatic downloads, extraction, labels, normalization |
| `train_gleason.pbs` | GPU | Smoke, full initial training, completed-stage resume |
| `evaluate_gleason.pbs` | GPU | Evaluation or frozen-teacher mining |
| `regions.pbs` | GPU | Legacy region grading; MODEL_DIR, REGION_INPUT, REGION_MPP |
| `stability.pbs` | CPU | Legacy stability analysis; PREDICTIONS, FOV_UM |
| `wsi.pbs` | GPU | Legacy WSI path; SLIDE, TUMOUR_MODEL, MODEL_DIR |

Region/WSI operations additionally require appropriate input material and a
compatible tumour-detection model. They are not prerequisites for either patch
training experiment. Set PATCH_SIZE explicitly for input resolution.

## Assumptions and limitations

The two tracks do not share teachers, training patches or mined examples.
NASNetLarge, Flatten/Dense(256)/three-output head, legacy staged unfreezing,
loss/Adam resets/flips and existing StainTools 2.1.2 adapter remain unchanged.
There is no stromal decontamination, invented combined-GP class, percentage-purity
threshold or subsequent expansion-training schedule.

The automatic Crowd/SICAP route uses raw Crowd images but **normalized SICAP**.
The source-normalization deviation is recorded. To use original SICAP, set all
four explicit source-root overrides in the config; only ImageNet weights are
then downloaded automatically. Pure seeds use matching patches from same-pattern
SICAP slides. Crowd majority-vote labels do not establish pure-source eligibility;
Crowd contributes to the separate `TRAINING_COHORT=published` baseline.
The original `wsi_labels.xlsx` must be supplied separately; it is not in the
normalized patch annotation archive.

Gleason2019 votes using all available expert masks at native resolution, including
background and benign. Missing masks abstain, value 6 abstains per pixel, and
unique plurality wins; ties remain unresolved. Unresolved pixels/fractions and
invalid-vote counts are recorded, without vetoing a whole patch/core. Cores with
no masks are audited and excluded. Tiles with no resolved tumour do not enter
supervised training; mixed-GP tiles carry no fabricated class target.

Single-pattern cores are a **consensus proxy**, not proven pure-pattern patient
cases. Eligibility considers the entire core, including discarded crop edges.
The approximately 80/20 seed-42 split is by filename slide group, frozen before
seed selection/mining; patient independence is unverified. Grouped proportions
may differ from 80/20 patch counts. Missing seed classes cause failure rather
than automatic split search. Full Gleason2019 evaluation uses its internal
validation partition; no labelled official test set is manufactured.

The 600→350 geometry restores the intended sampling scheme, but dataset,
annotation and cohort differences prevent claiming a resolution-only causal
comparison or exact reproduction of the published model. Full normalization,
GPU training and actual scheduler execution must be validated on the cluster.
See [the implementation report](../reports/gleason_implementation.md).

### Gleason2019 stain preprocessing failures

Gleason2019 preparation excludes individual tiles when the tile preprocessing
call raises an exception, including empty tissue masks, numerical errors, or
nonfinite normalized pixels. Each exclusion
emits a `RuntimeWarning` naming the core and tile coordinates in the CPU PBS log.
Valid tiles continue through the existing brightness/Macenko pipeline. No raw
image or expert mask is deleted, and no unnormalized fallback enters training.
The catch is limited to `preprocess_image(tile, normalizers)`. Reference fitting,
raw image decoding, mask handling, cache writes, and other operations outside
that call still abort on error. KeyboardInterrupt and SystemExit are not caught.
Review repeated errors: this broad tile policy can exclude tiles for a backend
problem as well as an unsuitable image.

Inspect `PREPARED_DIR/stain_exclusions.json` for each excluded tile's original
annotation status, class, split, coordinates, and error. `patch_inventory.csv`
retains its annotation fields with status `stain_failure`, empty cache fields,
and normalization error details. These rows are excluded from both training and
validation manifests, including the mixed-tile mining pool. `prepared.json` and
`gleason2019.json` record the exclusion count, policy, and audit-file hash.
Each exclusion is also appended immediately to `stain_exclusions.jsonl`, so
completed audit records survive a later interruption. The JSON summary and CSV
inventory are finalized on normal completion or a handled preparation error;
they may be absent or incomplete if the scheduler forcibly kills the job.
The original `single_gp`/`mixed` counts describe annotation categories before
normalization; `stain_failure` counts exclusions from those categories. Seed and
validation class counts describe retained manifest eligibility.

Core eligibility and the frozen split are unchanged. Preparation can be complete
with exclusions, but `training_ready` still requires all three seed GP classes
and nonempty validation after exclusions. Review the exclusion audit before
training: exclusions change the evaluated cohort and do not establish whether
the underlying annotations align with the images.

After replacing the code, use a fresh Gleason2019 `PREPARED_DIR` in `site.env`
(or remove only the stopped, incomplete prepared output after verifying its path).
Keep `DATA_ROOT/gleason2019`; its verified downloads are reused. Then submit once:

```bash
bash hpc/submit_pipeline.sh gleason2019 prepare
```

Review preparation warnings, `stain_exclusions.json`, and `prepared.json` before
submitting smoke or training. This exclusion policy applies only to Gleason2019
preparation; Crowd/SICAP behavior is unchanged.
