# Laptop group-equivariant training

Run from the repository root. The intended Python range is 3.11 through 3.16,
subject to dependency support; this is not a verified compatibility claim.
The current escnn workaround requires NumPy <2 and scikit-learn <1.2, whose
published wheels cover Python 3.11 but not the entire intended range. Python
3.11 remains the practical starting point. Supporting newer interpreters may
require a substantive escnn compatibility change, which has not been made.
These additions do not modify existing trainers. The model is copied unchanged from
`inbox/datasci-proj/group-equivariant-model.ipynb` in `ventidropbox/tavern-workspace`.
The new trainer uses all parameters from random initialization, cross-entropy,
Adam at 1e-4, and the notebook's ReduceLROnPlateau scheduler (factor 0.2,
patience 2). It defaults to 30 total epochs, batch size 4, no progressive
unfreezing, and no augmentation. Images use the notebook's RGB conversion,
bilinear 350x350 resize and [-1, 1] scaling. No stain normalization is added.

## Install

Create and activate a separate Python environment. Install matching PyTorch / torchvision wheels for your CUDA setup using the PyTorch install
instructions, then:

```sh
python -m pip install -r requirements-group-equivariant.txt
```

For a CPU-only environment, first install:

```sh
python -m pip install "torch>=2.5.1" "torchvision>=0.20.1" --index-url https://download.pytorch.org/whl/cpu
```

`escnn` depends on `py3nj`, which may require a Fortran compiler. WSL/Linux is
recommended on Windows if installation fails. See the upstream installation
notes: https://github.com/QUVA-Lab/escnn#installation . Keep the same environment
for resume. Nothing is installed or downloaded by the training script.

## Specify your data and start

Set `--data` to the directory containing the extracted original archive roots:

```text
/path/to/extracted/
  val_dataset_1_norm/  (original nested files)
  val_dataset_1_tu/    (original nested files)
```

Preserve their nested paths as recorded in `manifests/VAL1_manifest.csv`.
Do not point the script at archives or rearrange the extracted image files.
It checks the manifest fingerprint and that every expected image exists; it
assumes the image contents are those already verified for that manifest.
Keep checkpoints outside the image directory and outside Git, ideally on a
local disk with sufficient free space.

```sh
python -m modern_pca.train_group_equivariant --data "/path/to/extracted" --output "/path/to/checkpoints/eq-run"
```

The established `VAL1_internal_v1` membership is reused: 116,655 train patches,
29,164 internal-validation patches, seed 42/hash-ranked within class. Split
records are written under the output directory. VAL2 is never read. This is
a binary reimplementation experiment; the patch-level split does not establish
patient-level independence. The notebook's temporary VAL2 benchmark subset
is not used for training here.

`--device auto` chooses CUDA when available, otherwise CPU. Apple MPS is not
selected. `--batch-size 1` reduces memory use; choose this before starting.
`--epochs 30` is the total target, not 30 additional epochs on resume.

## Resume after an error or shutdown

Fix the cause (for example, free disk space or restore data availability), then
run the same command with `--resume`:

```sh
python -m modern_pca.train_group_equivariant --data "/path/to/extracted" --output "/path/to/checkpoints/eq-run" --resume "/path/to/checkpoints/eq-run/last.pt"
```

Keep the original batch size, model code and package versions. You can change
the data root if its relative paths and contents are unchanged. Only load your
own trusted checkpoints: these files include Python training state.

`last.pt` includes model parameters/buffers, Adam state, scheduler state,
random state, completed epochs, next batch and metrics. It is saved after
100 training batches **or** five minutes (whichever comes first, checked after
a batch), before validation, and after every completed epoch. Use
`--checkpoint-every 10` for more frequent batch checkpoints. A crash can lose
work since the last successful save; the saved batch order is reconstructed
and continues from the saved cursor. Interrupted validation restarts validation.

Each save writes a temporary file, flushes it, then atomically replaces
`last.pt`; a leftover `last.tmp` can be ignored. The previous complete file
remains available if writing its replacement fails. Hardware/kernel differences
can affect numerical results; recovery preserves training state, not a promise
of bitwise equality across devices. No automatic fallback changes the batch size.

At completion, `last.pt` is the final epoch model with optimizer state. Metrics
for completed epochs are in its `progress.history`; no best-epoch selection or
VAL2 evaluation is performed by this minimal trainer.

## Evaluate a fixed checkpoint on VAL2

Use the separate PyTorch evaluator with a trusted checkpoint saved after a
completed epoch. It preserves the trainer's preprocessing and argmax decision
rule (ties select benign), verifies the VAL2 manifest and every image SHA-256,
and loads the escnn checkpoint in training mode before switching to evaluation
mode. No training, augmentation, threshold tuning or checkpoint selection occurs.

```sh
python -u -m modern_pca.evaluate_group_equivariant \
  --data /path/to/extracted \
  --checkpoint runs/group_equivariant_run01/epoch1.pt \
  --output runs/group_equivariant_run01/val2_epoch1 \
  --batch-size 4 --device cuda
```

The extracted root must contain `val_dataset_2_norm/norm/` and
`val_dataset_2_tu/tu/`. The output directory must not already exist. Results are
`predictions.csv` and `metrics.json`, including tumour-positive precision,
recall, specificity, F1, ROC-AUC, accuracy, loss and the confusion matrix
(rows=true, columns=predicted; benign then tumour). Undefined metrics are null.
Only a completed `metrics.json` indicates success; `.partial` files are incomplete.
After a failed run, use a new output directory. Evaluation does not resume.

Report the evaluated checkpoint's completed epoch count. If VAL2 results inform
model changes or selection, VAL2 has participated in development and must no
longer be described as an untouched final test set.
