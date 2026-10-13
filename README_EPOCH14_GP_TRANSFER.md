# Binary epoch-14 -> few-shot Gleason transfer (Vanda)

This package creates exactly **nine headline conditions**: F44, F22, F11
(global-average-pooled) × {zero-fit cosine 1-nearest-neighbour,
5-shot class-mean prototypes, 50-shot class-mean prototypes}.

All neural weights remain frozen; the saved `epoch14.keras` is the original
**benign/tumour binary** classifier, NOT a Gleason model. The nine conditions
use one representation source, epoch14; an ImageNet-only encoder should be
added later if you want to attribute the advantage specifically to binary
pathology fine-tuning rather than generic ImageNet features.

To run this, edit site.env to point EPOCH14_MODEL towards epoch14.keras. 

### Preflight without GPU

```bash
cd "$REPO_DIR"
source "$PIPELINE_CONFIG"  # or specify the full site.env path
"$VENV_DIR/bin/python" hpc/epoch14_gleason_fewshot.py \
  --prepared "$PREPARED_DIR" --cache "$PREPARED_DIR/cache" --preflight
```

This checks full PANDA manifest integrity, image-content leakage, eligible
GP3/4/5 counts, and that every deterministic 5/50-shot support set is possible.
No model inference is run, and no files are written.

### Optional GPU model smoke test

Before extracting the full dataset, check that your Python/Keras environment
can load the exact 804 MiB checkpoint and extract all three GAP feature
vectors from one verified PANDA PNG. This costs one short PBS GPU job, and
writes no result artifacts:

```bash
# Temporarily add to your external site.env: PROBE_MODEL_SMOKE=1
# Submit the PBS script below, inspect its output for MODEL SMOKE PASS.
# Remove PROBE_MODEL_SMOKE or set it back to 0 before the full run.
```

### Submit
```bash
cd "$REPO_DIR"
export PIPELINE_CONFIG=/absolute/path/to/site.env
qsub -v PIPELINE_CONFIG="$PIPELINE_CONFIG" hpc/pbs/epoch14_gleason_fewshot.pbs
qstat -u "$USER"
```

Avoid submitting at the same time as the full Gleason GPU training if your
allocation permits only one running GPU job or has a tight allocation quota.

### Resume after PBS timeout

Feature extraction is streamed into `features/*.npy` memmaps, with a safe
last-complete-batch progress record. A job interrupted during extraction can
resume without rerunning completed batches. Result calculation can also be
restarted from the cached features.

Use the exact same absolute output path as the first job. Add **in the external
`site.env`**:

```bash
PROBE_OUTPUT_DIR=/scratch/USER/WORK/runs/panda-full/<FIRST_JOBID>-epoch14-gp-transfer
PROBE_RESUME=1
```

Then resubmit. Keep seeds, batch, manifest, checkpoint and script unchanged.

## Selection and inference protocol

1. Read SHA-256 verified `panda_train.csv` and `panda_validation.csv` from the
   frozen full-preparation directory. Restrict **both** to `region_role=pure`
   with matching grade codes, mirroring Gleason supervision.
2. Eliminate identical PNG contents within each split. If a hash has conflicting
   GP labels, eliminate every copy. Preserve a CSV audit of exclusions.
3. Check strict train/validation sample, content, WSI, and known-patient
   isolation. Missing patient IDs are explicitly not evidence of independence.
4. Choose nested 5/50-shot supports using manifest identities **before**
   extracting or looking at learned features. Prefer globally distinct WSIs,
   then within-class distinct WSIs, then reuse a WSI only if necessary.
   The sample IDs, filenames, class, rank, WSI, patient ID where present,
   coordinates, image SHA and selection tier are all exported.
5. Repeat over 20 deterministic seeds by default, with the **same support
   draws across F44/F22/F11**. The K=5 images are a subset of the K=50 images.
6. Extract all frozen embeddings once for each candidate train and validation
   patch, verifying each PNG against its frozen SHA by default. Input is
   `350x350 RGB uint8 PNG -> float32 / 255`, with NO extra normalization.
7. Normalize each GAP vector to unit L2 norm. Zero-fit uses exact cosine 1NN
   over **all eligible GP-labelled training embeddings**, never a fitted
   classifier. Predictions are determined by retrieved training labels.
8. The 5-/50-shot classifiers use only K labelled samples per class, normalized
   image embeddings, normalized per-class means, and cosine classification.
   No optimizer or cross-validation is applied.
9. Evaluate on the **same entire eligible validation population** in all
   nine headline conditions. Metrics: macro-F1, balanced accuracy, accuracy,
   per-GP precision/recall/F1, and confusion matrices.

**Terminology:** `zero_fit_1nn` is zero-*fit* retrieval, NOT genuinely zero-shot
Gleason classification. It uses the GP labels of the reference-image bank.

## Outputs

```
<JOBID>-epoch14-gp-transfer/
  run_config.json                    # immutable hash-locked protocol
  population_summary.json
  population_train.csv              # evaluated train bank IDs/provenance
  population_validation.csv         # evaluated query IDs/provenance
  excluded_rows.csv                 # content/label/region exclusions
  support_selection_audit.csv       # EXACT 50/class for every seed; flag 5/class
  support_diversity_summary.csv     # per-class unique WSI/patient counts
  model_load.json                   # TF version, checkpoint load mode
  features/
    identity.json
    train_progress.json
    validation_progress.json
    train_F44.npy, train_F22.npy, train_F11.npy
    validation_F44.npy, validation_F22.npy, validation_F11.npy
  prototypes/                       # each stage, 5-/50-shot, every seed
    F11_prototype_5_seed42.npy
    ...
  predictions/
    F44_zero_fit_1nn.csv            # every query + retrieved reference ID
    F44_prototype_5_seed42.csv      # every query + cosine-to-GP3/4/5 scores
    ...
  metrics_by_seed.csv
  metrics_detailed.json             # confusion matrices + full metrics
  summary_9_conditions.csv          # MAIN report table, nine rows
  summary_9_conditions.json
  result.json                       # completion/provenance/limitations
  artifact_sha256.json              # SHA256 and byte count for all artifacts
```

The 5-/50-shot methods have twenty per-seed results per stage, but are
aggregated into **six** rows in `summary_9_conditions.csv`; zero-fit adds
three rows. Every seed has its own prediction CSV and prototype array.

## Interpretative limits

- The output is **patch-level GP3/GP4/GP5**, not a WSI Gleason score.
- PANDA validation is internal held-out WSI data, not an independent external
  institution cohort, and missing patient IDs prevent patient-disjoint claims.
- The full labelled training bank used for nearest-neighbour lookup means the
  zero-fit condition has more reference information than 5- or 50-shot.
  Do NOT rank those three columns as if they had identical label budgets.
- This is a descriptive *epoch14-only* experiment. Comparison against frozen
  ImageNet NASNet F44/F22/F11 is needed for a controlled causal-pretraining
  advantage claim. Full-data Gleason training is a separate supervised bound.
- Cosine similarity scores are **not calibrated probabilities**.
- Selection prioritizes distinct slides, but the 50-shot cohort may need
  multiple patches per slide; `support_diversity_summary.csv` quantifies this.
- All explanations/metrics are research-only and do not establish a model
  explicitly recognizes gland categories or has clinical diagnostic validity.
