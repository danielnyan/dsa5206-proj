# ESCNN D8 / Stage-2G — VAL2 dependency + PBS evaluation pack

**Source basis:** `danielnyan/dsa5206-proj`, branch `extension-model-revised`, inspected head `00da1bf2604b5dfec5fe1cd4a717a0b55f1465eb` (23 Oct 2026).

The existing `modern_pca/evaluate_group_equivariant.py` already implements the Stage-2G-compatible `r.preprocess_legacy` pipeline. This pack deliberately does **not** modify it, the model, the trainer, or the selected checkpoint. It installs its missing dependencies into the established ESCNN venv in a CPU PBS job and uses PBS `afterok` to launch one fixed-checkpoint VAL2 GPU job.

## Included files

- `requirements-escnn-val2.txt` — pins only evaluation add-ons (`staintools`, `spams-bin`, `opencv-python`, `matplotlib`)
- `hpc/verify_escnn_val2.py` — checks the trained checkpoint's model/preprocessing contract; Macenko-processes one VAL2 image of each class
- `hpc/pbs/setup_escnn_val2_cpu.pbs` — installs pinned add-ons; freezes core torch/escnn/numpy/scipy/sklearn/Pillow versions; checks environment and real preprocessing
- `hpc/pbs/evaluate_escnn_d8_val2.pbs` — uses the existing evaluator with `best.pt` and batch size 4
- `hpc/submit_escnn_val2.sh` — submits CPU then GPU (`afterok`)

## Configure

1. Unzip into your **existing** `extension-model-revised` repository root. Do not overwrite training files, `best.pt`, `last.pt`, or Stage-2G cache.
2. In your EXISTING `/scratch/USER/site.env` (outside Git), set these values using actual paths:

```bash
ESCNN_VAL2_EXTRACTED="$WORK_ROOT/data/VAL2-extracted"  # must contain val_dataset_2_norm and val_dataset_2_tu
ESCNN_OUTPUT="$WORK_ROOT/runs/escnn-d8-stage2g-v1"  # existing finished run containing best.pt and last.pt
ESCNN_VENV_DIR="$WORK_ROOT/escnn-env"               # existing Python 3.11 PyTorch/ESCNN venv
# ESCNN_VAL2_OUTPUT="$ESCNN_OUTPUT/val2-stage2g-epoch7"  # optional; must NOT already exist
# ESCNN_PBS_PROJECT=personal-ACCOUNT               # only when required
# ESCNN_WHEELHOUSE=/shared/wheels                  # optional offline wheelhouse
```

`REPO_DIR` and `WORK_ROOT` remain as set by your current `site.env`. `ESCNN_SITE_SETUP` should load the correct Python/cluster modules as it did for D8 training. Do **not** set `ESCNN_VENV_DIR` equal to the Gleason environment `VENV_DIR`.

### Submit once from the login node

```bash
export PIPELINE_CONFIG=/scratch/USER/site.env
cd /path/to/your/extension-model-revised-checkout
bash hpc/submit_escnn_val2.sh
```

The script prints a CPU job ID, a dependent GPU job ID, and the exact result path. `qstat -u "$USER"` shows progress. Both jobs persist after terminal logout. `stdout`/`stderr` go to `"$WORK_ROOT/logs/escnn-val2"` using PBS `-o/-e`; some PBS clusters transfer logs only when jobs finish.

The CPU setup job tests actual stain normalization and checkpoint contract, not merely `import staintools`. If it fails, GPU evaluation does not start. It *preserves* PyTorch, ESCNN, NumPy, SciPy, scikit-learn, and Pillow versions. It does not recreate the venv or touch any checkpoint, source image, or original Stage-2G cache. `pip freeze` before and after is retained under the log directory.

Successful GPU output:

- `metrics.json` (contains `metrics.accuracy`, `metrics.recall_sensitivity`, `metrics.specificity`, `metrics.f1`, `metrics.roc_auc`, `metrics.FP`, `metrics.FN`)
- `predictions.csv` (all 34,103 VAL2 images)

The output directory is unique by default and never overwritten. An interrupted evaluation is NOT resumable: it retains its partial output for diagnosis, and a new submission uses a different output directory. Do not point the evaluator at an existing attempt directory.

## Scientific and operational caveats

- VAL2 has been examined in earlier project experiments and is not newly researcher-blind.
- The Stage-2G reference picture and StainTools Macenko procedure are reused; the **exact** original Stage-2G package versions may differ. For strict pixel parity, cross-check with a few cached VAL1 patches before asserting identical numerical preprocessing.
- VAL2 inference presently preprocesses images **serially inside the GPU job**. This is scientifically faithful to the revised repository, but can leave GPUs underutilized and consumes GPU allocation while CPU processing occurs. An independent CPU-preprocess/GPU-infer refactor would need validation and is intentionally not slipped into a dependency-only patch.
- Pip wheels and library versions may be unavailable if compute nodes cannot reach PyPI. Supply an offline `ESCNN_WHEELHOUSE` if available. The `spams-bin` binary may need compatible system OpenBLAS/GFortran runtime libraries; the CPU preflight catches import failures.
- If CPU setup fails, inspect the PBS logs. Do not install `requirements-reimplementation-wsl.txt` into the ESCNN environment or rebuild the successful training environment.
