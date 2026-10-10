# ENS model 2 (500 px NASNetLarge) on NSCC ASPIRE 2A

The second convnet of the paper's ENS strategy (Tolkach et al. 2020, Fig. 2d): the same
NASNetLarge recipe as model 1 but on 600 px patches resized to 500 px (about x35 instead of x23).
Code: `ens_model_training/ens_cache.py` (500 px uint8 cache from the MD5-verified VAL1 ZIPs) and
`ens_model_training/ens_train.py` (resumable training, epoch-14 export, grey-zone ENS rule
`ens_combine`). Trained on the frozen VAL1 split (116,655 train / 29,164 internal validation).
VAL2 is never downloaded or read by anything here.

Recipe (frozen, identical to model 1 except the input size): ImageNet weights, Flatten,
Dense(256, ReLU), Dense(2, softmax); brightness standardisation + Macenko stain normalisation,
flips; 14 epochs, epochs 1-7 head + top two cells at LR 1e-5, epochs 8-14 release cells
14/11/9/7/5/3/1 at 1e-6, Adam reset per stage; frozen BatchNorm; float32; physical batch 4 with
gradient accumulation to an effective batch of 100; seed 42; no model selection (epoch 14 is final).

## Results (NSCC, one A100-40GB, 2026-10-09/10, 26.6 GPU-hours)

Internal validation after each epoch (accuracy, %):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Acc | 94.80 | 95.75 | 95.96 | 96.12 | 95.68 | 96.39 | 96.27 | 96.73 | 96.86 | 96.95 | 96.99 | 97.01 | 97.16 | 97.11 |

Epoch 14 on the 29,164 internal-validation patches (10,562 tumour, 18,602 benign; tumour iff
p > 0.5):

| Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC | Balanced acc |
|---|---|---|---|---|---|---|
| 97.11% | 96.07% | 95.94% | 97.77% | 96.01% | 0.9943 | 96.86% |

TP 10,133 / FN 429 / TN 18,188 / FP 414. The curve has the shape of the paper's Fig. 2a: a plateau
while only the head trains, then a second rise once convolutional cells are released. The paper
reports no standalone model-2 number; model 2 is used only through the ENS rule (paper: VAL1 native
96.2% -> ENS 96.4%, VAL2 native 96.7% -> ENS 97.0%, gain mostly in tumour sensitivity).

Artifacts (not in Git): `ens_model2_epoch14.keras` (1.40 GB, SHA-256
`49305a192b3dfbdad7c9cea641f4195ae096389df0655863022c01f0d58a4680`), `metadata.json`
(recipe, software versions, checksums), `model2_internal_validation_predictions.csv`
(sample_id, label, p_tumour), `history.json`, `train.log`.

## Running it

```sh
source ens_model_training/jobs/config.sh                      # set NSCC_PROJECT and NSCC_ROOT first
bash ens_model_training/jobs/setup_env.sh                     # login node: conda env, tensorflow[and-cuda], pins, NASNet weights
qsub -P $NSCC_PROJECT -q $CPU_QUEUE -v CODE_DIR=$PWD ens_model_training/jobs/run_tests.pbs
qsub -P $NSCC_PROJECT -q $CPU_QUEUE -v CODE_DIR=$PWD ens_model_training/jobs/download_val1.pbs   # or bash on the login node if compute has no internet
qsub -P $NSCC_PROJECT -q $CPU_QUEUE -v CODE_DIR=$PWD ens_model_training/jobs/cache_parts.pbs      # 8 array tasks, ~15 min each on 32 cores
qsub -P $NSCC_PROJECT -q $GPU_QUEUE -v CODE_DIR=$PWD ens_model_training/jobs/gpu_probe.pbs        # optional: img/s + memory at batch 4 and 2
bash ens_model_training/jobs/submit_train_chain.sh 3                                               # 3 chained 24 h GPU jobs
```

Each training link checkpoints every 30 min, stops 30 min before its walltime, resumes from the
newest checkpoint, and exits at once when `status.json` says `COMPLETE`. Storage on scratch: ZIPs
12.4 GB, cache 109 GB, run directory ~8 GB plus the 1.4 GB export.

Ensemble on VAL1 internal validation (needs model 1's `epoch14.keras` and its sidecar):

```sh
qsub -P $NSCC_PROJECT -q $GPU_QUEUE -v CODE_DIR=$PWD,MODEL1=/path/epoch14.keras,MODEL1_META=/path/metadata.json,MODEL2_RUN=ens_model2_b4 ens_model_training/jobs/ensemble.pbs
```

## NSCC notes

- `activate_env` exports `LD_LIBRARY_PATH` to `site-packages/nvidia/*/lib`; without it TensorFlow's
  pip CUDA wheels fail with "Cannot dlopen some GPU libraries". `staintools` is installed `--no-deps`,
  so `matplotlib` is installed explicitly.
- GPU jobs with walltime <= 2 h route to `gdev`, longer ones to `g1..g4`; CPU jobs to `q*`.
- Login-node internet is throttled (PyPI ~25 KB/s, Zenodo ~80 KB/s); `aria2c -x16` gets 1-5 MB/s.
  Start long login-node processes with `setsid nohup ... < /dev/null &` and keep the shell in `$HOME`,
  since `/scratch` stalls interactive shells.
- Reading the cache through NumPy memmaps gave ~7 img/s on Lustre; `Model2Cache.get` does one
  contiguous read per tensor and `ens_train.batches` keeps 8 reads in flight (~33 img/s).
