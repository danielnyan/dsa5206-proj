# Genuine 5-/50-shot linear probing (frozen epoch14.keras features)

This is a **CPU-only** follow-up to the complete `epoch14_gleason_fewshot.py` run. It reads the original, immutable feature arrays and audited 5-/50-shot support sets, rather than running NASNetLarge again. For each F44/F22/F11 × 5/50 shots × 20 seeds, it trains a **multinomial L2-regularized logistic regression with learned weights and intercepts**. Unlike cosine prototypes, this is a genuine fitted linear probe.

**Required existing files in `PROBE_DIR`:** `run_config.json`, `result.json`, `metrics_by_seed.csv`, `support_selection_audit.csv`, `population_train.csv`, `population_validation.csv`, `features/identity.json`, `features/train_progress.json`, `features/validation_progress.json`, `features/{train,validation}_{F44,F22,F11}.npy`. The original 8.4 GiB feature bank must still be present on Vanda. The ZIP uploaded for results review does **not** contain the `.npy` feature arrays.

## Install

From the extracted ZIP, copy `hpc/epoch14_gleason_linear_probe.py` to `$REPO_DIR/hpc/` and `hpc/pbs/epoch14_gleason_linear_probe.pbs` to `$REPO_DIR/hpc/pbs/`. There is no need to edit the existing GP-probe, its cache or checkpoint.

```bash
export PIPELINE_CONFIG=/scratch/<USERNAME>/site.env
source "$PIPELINE_CONFIG"
export PROBE_DIR="$RUN_ROOT/<OLD_GP_PROBE_JOBID>-epoch14-gp-transfer"

# Ensure the old run still contains all feature files.
ls -lh "$PROBE_DIR/features"/{train,validation}_{F44,F22,F11}.npy

# Fast metadata/feature shape preflight on login node; no model or GPU.
cd "$REPO_DIR"
"$VENV_DIR/bin/python" -m hpc.epoch14_gleason_linear_probe \
  --source "$PROBE_DIR" --output "$RUN_ROOT/linear-preflight-unused" --preflight

# PBS without a GPU. Pass PROBE_DIR via qsub -v, or set PROBE_DIR in site.env.
qsub -v "PIPELINE_CONFIG=$PIPELINE_CONFIG,PROBE_DIR=$PROBE_DIR" \
  hpc/pbs/epoch14_gleason_linear_probe.pbs
```

The job uses `ncpus=8`, `mem=32gb`, `walltime=06:00:00`, **no GPU**. Include `#PBS -P personal-...` if your cluster requires a project/account directive. Check queued PBS output for `PREFLIGHT PASS`, `FIT`, `PREDICT`, `SUCCESS`.

By default results are written to `$RUN_ROOT/<NEW_PBS_JOBID>-epoch14-linear-probe/`. The original GP-probe output is never modified.

## Outputs

- `summary_6_conditions.csv`: mean ± SD macro-F1, balanced accuracy over the same 20 seeds; paired mean improvement over old prototypes.
- `metrics_by_seed.csv`: full per-seed metrics, per-class precision/recall/F1, and paired comparison with prototypes.
- `metrics_detailed.json`: all confusion matrices (120 trained linear models).
- `predictions/F44_labels.npy`, `F22_labels.npy`, `F11_labels.npy`: all validation predictions for every (shots,seed), in order **5-shot seeds 42–61, then 50-shot seeds 42–61**.
- `predictions/<stage>_linear_{5,50}_seed42.csv`: auditable validation predictions for the first seed.
- `run_config.json` and `result.json`: provenance and completion status.

To inspect results:

```bash
find "$RUN_ROOT" -maxdepth 2 -name summary_6_conditions.csv -print
cat "$RUN_ROOT/<NEW_PBS_JOBID>-epoch14-linear-probe/summary_6_conditions.csv"
```

## Protocol and limitations

- Binary epoch14.keras stays frozen. Features are from the original probe, no new GPU inference.
- Training uses precisely 5 or 50 **labelled** PANDA images **per class**, selected from audited WSI-diverse sets, for each of 20 seeds. No validation labels used for fitting or selecting the hyperparameter.
- Input embedding rows are L2-normalized; logistic regression includes three fitted class weights and intercepts and uses L2 regularization with **fixed prespecified C=100**, `lbfgs`, 1000 max iterations. `C` is configurable via `LINEAR_PROBE_C`; do not choose it after checking validation results. This implementation deliberately does not fit a support-dependent scaler or PCA.
- Validation split is internal held-out PANDA, not independent external testing. Source identities prove WSI and exact-image-content separation, not patient separation.
- No second encoder or ImageNet-only feature baseline is introduced. Results evaluate linear decodability of the *existing binary checkpoint*, not the impact of its fine-tuning.
- Prototype comparison is paired on exact support images; high-dimensional logistic regression can overfit 5-shot supports, so macro-F1 and per-class errors should be evaluated empirically.
- Optional sensitivity runs with `LINEAR_PROBE_C=1` or `10` must use distinct output directories; interpret as sensitivity analysis, not validation-driven model selection.
