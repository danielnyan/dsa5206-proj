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

## Train to a fixed stopping rule, then evaluate once

`modern_pca.run_group_equivariant_experiment` controls the existing trainer and
evaluator. It uses the **stopping epoch**, not the best historical epoch, for
the reported VAL2 result. The model, preprocessing, Adam optimizer and learning
rate scheduler are unchanged.

The default rule is deliberately a **joint stability rule**:

| Setting | Default |
|---|---:|
| Maximum total epochs | 30 |
| Minimum completed epochs before early stopping | 10 |
| Window of consecutive completed epochs | 4 |
| Maximum range of training loss within the window | 0.001 |
| Maximum range of validation loss within the window | 0.001 |
| Maximum range of training accuracy within the window | 0.001 (0.1 percentage points) |
| Maximum range of validation accuracy within the window | 0.001 (0.1 percentage points) |

All four ranges (`max - min`) must be at or below their tolerances for early
stopping. A single small adjacent change is insufficient. The maximum epoch
limit stops the run even if stability is never reached. Unlike validation-loss
patience stopping, this rule does not stop merely because validation worsens
while training improves. No best-weight restoration or VAL2-based selection is
performed. The existing `ReduceLROnPlateau` still monitors VAL1 validation loss.

These are starting tolerances chosen from the VAL1 pilot, not statistically
estimated optimal thresholds. Pilot evidence:

| Epoch | Training loss | Validation loss | Training accuracy | Validation accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.154626 | 0.096735 | 94.1520% | 96.6294% |
| 2 | 0.079125 | 0.100067 | 97.1969% | 96.1528% |
| 3 | 0.062985 | 0.055856 | 97.7772% | 98.0730% |
| 4 | 0.053746 | 0.048336 | 98.1141% | 98.4159% |
| 5 | 0.046490 | 0.047411 | 98.4116% | 98.4570% |

Epoch 4 to 5 changed training loss by about 0.007256 and validation loss by
0.000925. Training accuracy changed by 0.2975 percentage points and validation
accuracy by 0.0411 points. This is not yet joint stability. Ten minimum epochs
provide additional observation beyond the five pilot epochs; thirty retains
the original training cap. The four-epoch window reduces sensitivity to a
single quiet epoch and has room to observe the existing scheduler's changes.

From the repository root, inspect the protocol without writing or training:

```sh
python -u -m modern_pca.run_group_equivariant_experiment \
  --data /home/chenlehan/Desktop/5206/data/zenodo_3825933/extracted \
  --output runs/group_equivariant_stability01 \
  --resume-from runs/group_equivariant_run01/epoch5.pt \
  --plan
```

Remove `--plan` to start. This copies the pilot checkpoint and continues from
epoch 6 in the new directory, leaving the pilot files intact. To run from random
initialization instead, omit `--resume-from`. Optional overrides are
`--max-epochs`, `--min-epochs`, `--window`, `--loss-tolerance`,
`--accuracy-tolerance`, `--batch-size`, `--checkpoint-every` and `--device`.

After an interruption, use the **same settings** and `--resume`, without
`--resume-from`:

```sh
python -u -m modern_pca.run_group_equivariant_experiment \
  --data /home/chenlehan/Desktop/5206/data/zenodo_3825933/extracted \
  --output runs/group_equivariant_stability01 \
  --resume
```

The protocol, code fingerprints and environment versions are frozen at startup;
changed settings/code/environment require a separate experiment. A process lock
prevents two controllers writing the same experiment. The original trainer's
optimizer, scheduler, RNG and within-epoch cursor recovery remain in use.

Artifacts in the experiment directory:

- `protocol.json`: frozen settings, code/environment fingerprints and pilot provenance.
- `initial.pt`: immutable pilot snapshot when starting from a checkpoint.
- `training/last.pt`: recovery checkpoint; `training.log`: appended training output.
- `epochs/epoch_NNN.pt`: each newly completed epoch's checkpoint.
- `history.csv`: all completed epochs; `stopping_status.json`: current window ranges.
- `final.pt` and `selection.json`: fixed stopping checkpoint and exact stopping reason.
- `val2/attempt_NNN/metrics.json` and `predictions.csv`: successful full VAL2 evaluation.
- `summary.json`: final VAL1 history, VAL2 metrics, checkpoint selection and artifact hashes.

VAL2 images are only read after checkpoint selection. An interrupted evaluation
is retried from the beginning in a new attempt directory; partial attempts are
retained. A completed evaluation is reused on resume, including a crash between
evaluation completion and summary export. Training subprocess errors never
trigger VAL2 evaluation.

VAL2 has already been explored at earlier checkpoints in this project. The
protocol and final report explicitly record this fact. This workflow makes the
new stopping rule reproducible; it does not retroactively make VAL2 an untouched
test cohort. Do not change the rule after seeing its resulting VAL2 metrics.
