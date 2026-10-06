# Tumour-only Gleason training on PBS

This implementation adapts the released Tolkach training code to labelled
CrowdGleason and SICAPv2 tumour patches. It uses published Crowd `MV`, published
SICAP patch classes, removes NC, and outputs `GP3,GP4,GP5`. There are now two
explicit initial-training cohorts: `pure` (default, for subsequent mining) and
`published` (the original combined Crowd+SICAP labelled baseline). Validation/test
partitions retain their supplied membership; no mixed Gleason score is an output
class.

SICAP's `wsi_labels.xlsx` contains **slide-level** `Gleason_primary` and
`Gleason_secondary`, not two independent patch targets. Preparation keeps these
separate from the published patch label. The pure cohort uses tumour patches
from 3+3, 4+4 or 5+5 slides only when the patch class matches that GP. Multipattern
slides are reserved as mining candidates, not labelled wholesale by their
primary pattern. Unknown/conflicting source annotations remain in the manifests
for audit and the published baseline, but not the pure seed cohort. Crowd `MV`
alone does not establish pure-region provenance, so Crowd is currently included
only in the published baseline. Same-pattern slide grades are a source-backed
proxy, not proof of the original study's pathologist-annotated unequivocal purity.

The exact architecture is ImageNet NASNetLarge, 350x350 RGB, Flatten,
Dense(256, ReLU), Dense(3, softmax). The 14-epoch schedule, categorical loss,
Adam resets, horizontal/vertical flips, and trainable BatchNorm after each
release boundary follow `1_training/NASNetLarge_350px_FULL_training.py`.
The existing `modern_pca.models.build_classifier` now has an optional
`backbone_training` parameter: other callers retain their previous `False`;
the Gleason trainer uses `None` to propagate training/evaluation mode.

StainTools 2.1.2 and the existing compatibility adapter in `evaluate_paper.py`
provide brightness/Macenko preprocessing. The separately implemented NumPy
normalizer is not used. Cache PNGs preserve uint8 normalized pixels exactly.

No stromal decontamination, invented mask/consensus filtering, new subpatch labels,
GAP head, class weighting, or new scoring rules are added. Optional confidence
mining follows the paper's 0.95 rule; it does not invent subsequent retraining
epochs or an iterative curriculum. Original training material is unavailable.
Whole public patches retain their source labels and are resized to 350x350;
their field of view/resolution differs from the original approximately 150 um
training fields. Mined labels are exported separately with their original labels.

## Configure the cluster

Copy the extracted repository to shared scratch. Edit account, queue, GPU
resource syntax, and walltime in `hpc/pbs/*.pbs` to match your PBS installation.
The headers use PBS Pro/OpenPBS `select`; other PBS variants may need edits.
Resource requests are starting points; GPU VRAM is site-specific and must be
selected through the site's resource syntax. The full Flatten model is about
210M parameters. Batch 100 may require substantially more memory than your GPU;
measure batch 1 first. A smaller production batch changes BatchNorm statistics
and is recorded as a deviation. There is no automatic batch/precision fallback.

```bash
cd /scratch/USER/dsa5206-proj
cp hpc/site.env.example /scratch/USER/gleason-site.env
# Edit all paths and SITE_SETUP. Reserve at least ~100 GB scratch initially.
export PIPELINE_CONFIG=/scratch/USER/gleason-site.env
bash hpc/bootstrap_env.sh
```

Bootstrap uses the repository's pinned TF/Keras/native dependency environment
plus OpenPyXL/OpenSlide. Use cluster-provided compatible modules if appropriate.
If your site prohibits compilation/package installation on login nodes, perform
bootstrap on an interactive CPU allocation with internet access.

## Download before scheduling compute

On the permitted login or transfer node:

```bash
bash hpc/download_data.sh /scratch/USER/dsa5206-data raw-crowd
```

Downloads have resume/retry and published MD5 checks. Sources are
[CrowdGleason Zenodo v1](https://zenodo.org/records/14178894) and the official
Keras NASNetLarge no-top ImageNet weights. The downloader stages weights in
`DATA_ROOT/keras/models` so training does not need outbound network access.
About 17 GB of archives plus 0.35 GB of weights must be staged before extraction.
Sources, extracted images, caches, and training output stay outside the repo.

The stable automated route supplies **normalized SICAPv2**, while `raw-crowd`
selects unnormalized Crowd images. `ALLOW_PRENORMALIZED=1` explicitly accepts
this release and records that the legacy Tolkach normalization is applied after
the source's normalization. This is a preprocessing limitation. Set the four
explicit source paths in `site.env` to use original SICAP images instead; masks
are not needed. The original SICAP layout uses
`partition/Validation/Val1/{Train,Test}.xlsx` for train/validation and
`partition/Test/Test.xlsx` for held-out test. No invented random split is used.
The original dataset landing page is
[SICAPv2 v1](https://data.mendeley.com/datasets/9xxm58dvs3/1); its legacy bulk URL
was inaccessible during implementation, so the script does not depend on it.

For `TRAINING_COHORT=pure`, stage the original `wsi_labels.xlsx` separately and
set `SICAP_WSI_LABELS` in your site configuration. The normalized annotation ZIP
contains only Train/Val/Test workbooks, not slide grades. Preparation can discover
`wsi_labels.xlsx` under an explicit SICAP annotation root; otherwise provide its
path. It records the workbook SHA-256 and patient IDs. Missing metadata never
falls back to assuming a pure GP. For the published-label baseline, use
`TRAINING_COHORT=published` and leave `SICAP_WSI_LABELS` empty if unavailable.

## Submit preparation, then a GPU smoke run

```bash
PIPELINE_CONFIG=/scratch/USER/gleason-site.env MODE=smoke bash hpc/submit_pipeline.sh
```

Preparation runs only in the CPU job. It authenticates archives, checks safe ZIP
paths, extracts them, joins labels, audits every available image (including NC),
removes NC, and builds the derived cache/manifests. `prepared.json` is written
only after successful completion. Missing image joins fail production. Duplicate
image content crossing splits fails; shared slide prefixes are reported in
`audit.json` while supplied membership is preserved. Slide prefixes alone do not
establish patient independence. Preprocessing workers are bounded; each uses the
same fixed stain reference.

Use a new `PREPARED_DIR` for this revision: older manifests lack source-grade
metadata and cannot support the pure/mining route. Inspect `region_role_counts`
in `audit.json` before training. Pure training and mining abort on known overlap
with held-out samples, image content, slide IDs or patient IDs; they do not
silently discard rows or invent a new split. Resolve any such overlap explicitly.

The smoke job reads only train/validation caches, limits samples, and runs one
epoch at boundaries 17 and 1. It checks finite operations, classifier updates,
a frozen weight, and final model reload. Smoke models cannot be used for final
test evaluation or mining. All three GP classes must exist in the selected
training cohort. The repository includes no trained checkpoint.

Inspect the smoke result and PBS resource accounting. Adjust `BATCH_SIZE` and
resource requests. Production defaults to the legacy physical batch 100.

```bash
PIPELINE_CONFIG=/scratch/USER/gleason-site.env SKIP_PREP=1 MODE=production bash hpc/submit_pipeline.sh
```

Use a new `PREPARED_DIR` when changing source data or preprocessing. The CPU
preparer refuses to overwrite an existing output directory. An interrupted
extraction can be rerun; derived output should use a new directory.

Each completed training stage writes an alternating recovery checkpoint plus a
checksum-bound `checkpoint.json`. Resume uses the last completed stage; a stage
interrupted before checkpoint completion is repeated. Adam resets at every stage
as in the legacy script. This is not a claim of bitwise restart equivalence.

```bash
qsub -v PIPELINE_CONFIG=/scratch/USER/gleason-site.env,MODE=production,RESUME=1,MODEL_DIR=/scratch/USER/dsa5206-runs/JOB-production hpc/pbs/train_gleason.pbs
qsub -v PIPELINE_CONFIG=/scratch/USER/gleason-site.env,MODEL_DIR=/scratch/USER/dsa5206-runs/JOB-production hpc/pbs/evaluate_gleason.pbs
```

Evaluation verifies the final checkpoint and frozen manifest/cache hashes and
reports tumour-only per-class precision/recall/F1, confusion matrix, accuracy,
and Cohen's kappa separately for each dataset. Raw probabilities are exported.
Use `MODE=smoke` in the evaluation job for a bounded validation-only check.
Neither training nor smoke evaluation reads test cache images; CPU preparation
does apply fixed, reference-based preprocessing to all supplied partitions.

## Mine multipattern training patches with the frozen initial model

After the full **pure-cohort** initial run has completed:

```bash
qsub -v PIPELINE_CONFIG=/scratch/USER/gleason-site.env,MODE=mine,MODEL_DIR=/scratch/USER/dsa5206-runs/JOB-production hpc/pbs/evaluate_gleason.pbs
```

This reuses the existing GPU inference job, invoking
`python -m modern_pca.evaluate_gleason --mine` with the model, manifests, cache and
a new output directory. No scheduler job is automatically launched after mining.

Only documented multipattern **training** rows are inferred. The immutable
initial model must be complete, non-smoke, trained with `training_cohort=pure`,
and bound to exactly the same manifests and source-grade workbook. The published
baseline and older models without cohort provenance cannot serve as teachers.
Validation/test metadata is read solely for exclusion/overlap checks; their
images and probabilities never enter selection. BatchNorm is in inference mode.

Retain a patch only if a single unrounded GP softmax probability is **strictly
greater than 0.95**. Exactly 0.95 is rejected. Its mined target is that GP, not
3+4, 4+5, the source primary grade, or a restriction to the source pair. Outputs:

- `candidate_predictions.csv`: all candidates, three probabilities and decisions.
- `mined_train.csv`: retained rows, mined GP, original annotation, source/cache
  identity and teacher checksum. Empty selections still have valid headers.
- `mining.json`: frozen input hashes, threshold, class counts and artifact hashes.

The scientific basis is Tolkach et al., Methods, “Development of biologically
meaningful Gleason grading algorithm/training”
([paper](https://doi.org/10.1038/s42256-020-0200-7)): initial pure-pattern data is
extended using high-confidence predictions on multipattern sources. The available
methods text and released training script do not specify a complete subsequent
retraining recipe (weight initialization/continuation, epoch count, mixture or
repeated rounds). Accordingly this implementation **stops at an audited expansion
pool**. It does not silently train on that pool or claim a reproduced expanded
model. Decide and document the supported subsequent procedure before consuming it.

## Folders 3, 4 and 5

`modern_pca.gleason_regions validate` executes the legacy large-image tiling,
preprocessing, and soft-probability summation. Its default offsets match folder
3; `--center-crop` uses the centered grid from folder 5. Predictions can be reused
for folder 5 without repeated inference. `legacy_score` extracts and executes
the original pure `gscoring()` function; no model is loaded from the old script.

```bash
python -m modern_pca.gleason_regions validate --model-dir RUN --input IMAGES --output REGION_OUTPUT --mpp 0.25 --patch-size 600 --center-crop
python -m modern_pca.gleason_regions stability --predictions REGION_OUTPUT/predictions.csv --output STABILITY_OUTPUT --fov-um 150
```

Run those commands through `hpc/pbs/regions.pbs` (GPU) and `stability.pbs` (CPU),
setting the required variables documented in the files. Stability preserves
sampling without replacement and 20 rounds, clips requests to available tiles,
and skips regions with fewer than 16 tiles. Default 1-16 matches the paper;
`--max-tiles 19` matches the released loop. Record actual physical field size.
Pixel dimensions alone cannot establish the original sampling-area comparison.

`modern_pca.gleason_wsi` ports the legacy C1/C8/environment and Type-2 path.
Use `hpc/pbs/wsi.pbs` with `SLIDE`, `TUMOUR_MODEL`, and `MODEL_DIR`. The default
gate is the legacy three-class tumour model (index 2); the existing constrained
binary gate can be explicitly selected with `TUMOUR_CLASSES=2,TUMOUR_INDEX=1`.
It must use the same normalized 350x350 `/255` input convention.

The WSI wrapper preserves tissue/nucleus thresholds, grey zone `[0.2,0.8)`,
eight-view classwise medians, native `>0.5` tumour decisions, and the environment
check's raw resize-and-scale input. Bounds checks fix negative-index wraparound
and out-of-range neighbours. Environment-disabled execution is valid. No-tumour
slides report no grade. The legacy +1 detection-grid offset, aligned grading grid,
rounded patch percentages, and nonempty-slide pseudocounts remain documented
behaviour. Heatmap generation uses direct colours and reports the same class
semantics; it does not duplicate the old decorative bitmap tiles.

Set `PATCH_SIZE` explicitly for the source resolution; default is legacy 600
level-0 pixels. Slide MPP is recorded, not silently inferred. Test source-domain
and FOV suitability before interpreting grades from replacement-data models.
Whole-slide execution still needs suitable slides and a compatible tumour model.

## Local validation and release

See `reports/gleason_implementation.md` for checks actually run. Full GPU
training, GPU serialization, WSI model inference, and PBS scheduler execution
remain cluster validation steps. The small VM has no provisioned GPU or full
datasets. The sample-only preparation flag permits structural checks and is
rejected by production training and final evaluation.

The ZIP contains source, original examples, existing reference manifests/reports, and these
jobs, with per-file SHA-256 in `RELEASE.json`. It excludes Git internals, secrets,
site configuration, environment packages, caches, training runs, and data archives.
# Automatic Gleason2019 preparation

After the usual environment bootstrap, set `DATASET=gleason2019` in your
absolute `site.env` file and choose separate raw `DATA_ROOT` and new
`PREPARED_DIR` directories on shared scratch. Submit from the login node:

```bash
qsub -v PIPELINE_CONFIG=/absolute/path/site.env hpc/pbs/prepare_gleason.pbs
```

The existing CPU PBS job downloads all official public training JPEGs and the
SHA256-pinned public annotation mirror, safely extracts all six expert archives,
cuts aligned non-overlapping native 600x600 image/mask tiles, then uses the
existing 350x350 Lanczos/brightness/Macenko preprocessing. No GPU is required.
Compute nodes need outbound HTTPS (Sync and Mendeley). If networking is blocked,
arrange the cluster's approved transfer/network-enabled compute queue first;
there is no login-node heavy-processing fallback. Allow raw/downloaded data plus
normalized-cache space on scratch. Downloads are atomic and checksum-resumable;
prepared output must be new, so interrupted preparation needs a new output path.
Preparation currently processes one core at a time to bound memory; `PREP_WORKERS`
does not parallelize the Gleason2019 path.

`patch_inventory.csv` audits every full tile, including excluded tiles;
`cache/` contains only resolved tumour-containing tiles, and `gleason2019.json`
records provenance and completion. All available expert labels, including background, vote per
native pixel: a unique plurality is retained, ties and any raw value 6 remain
unresolved. Tiles with unresolved pixels are excluded conservatively; no raw-6
remapping, contour correction, purity threshold, or stromal decontamination is
introduced. A single observed GP is a *patch annotation*, not evidence of a
pure 3+3/4+4/5+5 case. Mixed tiles are cached without a class target. Incomplete
edge tiles are discarded and counted. Masks are never interpolated for voting.

This is deliberately an **unsplit preparation corpus**. Official held-out masks
and verified patient/pure-case metadata are not supplied here. Do not feed this
inventory directly to the existing training/mining commands: agree leakage-safe
splits and teacher eligibility first. The submit helper queues preparation only
for this dataset, protecting against accidental baseline/pure-model training.
