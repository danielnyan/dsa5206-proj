# DSA5206 — Stage 2J / Epoch-15 Checkpoint Inventory

## Overview

This checkpoint preserves the code, configuration, metadata, and
evaluation results for the following NASNetLarge experiments:

1. Vanda epoch-14 baseline.
2. SICAPv2 full fine-tuning (epoch 15).
3. SICAPv2 head-only fine-tuning (epoch 15).
4. SICAPv2 partial fine-tuning (epoch 15).
5. Singapore dataset fine-tuning (epoch 15).

All epoch-15 models were initialized independently from the same
Vanda epoch-14 model.

## Model checkpoints

All paths below are relative to:

`/scratch/e1536052/DSA5206/`

| Experiment | Checkpoint path | SHA-256 |
|---|---|---|
| Vanda epoch 14 | `vanda_runs/nasnet_production_14epoch/epoch14.keras` | `433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde` |
| SICAPv2 full epoch 15 | `vanda_runs/sicap_epoch15_vanda_2xa40/epoch15.keras` | `09c3427f4ef37d3674e5bdc6d098fc1aec02d23e16342e5158cc4bd00e5b6521` |
| SICAPv2 head-only | `vanda_runs/sicap_head_only_epoch15/epoch15_head_only.keras` | `4d5b2127484f1e2c37ff33c3bd7ca451f77984d47b4f43b09f9710c6fd64dfd7` |
| SICAPv2 partial | `vanda_runs/sicap_partial_epoch15_lr1e7/epoch15_partial.keras` | `f28c7cbd6b874a4287972913127a361d7bbc8d2bcd8cf03c72747d8589c707df` |
| Singapore epoch 15 | `singapore_external/results/epoch15_singapore/full11020/epoch15_singapore.keras` | `5c1bf2f62c3fdb44a1dba239afa8718ab76a4d6b8b7be8b5195666ef3bc9dea8` |

The large model binaries and optimizer state are retained on Vanda
and are not committed to ordinary Git.

The model hashes above have been checked against the available
training metadata and prior integrity-verification results.

## Archived evaluation results

| Experiment | Local directory |
|---|---|
| Vanda epoch 14 VAL2 | `experiments/epoch15/baseline_epoch14/val2_results/` |
| SICAPv2 epoch 15 VAL2 | `experiments/epoch15/sicapv2/val2_results/` |
| Singapore epoch 15 VAL2 | `experiments/epoch15/singapore/val2_results/` |
| SICAPv2 development and VAL1 retention | `experiments/epoch15/sicapv2/followup_results/` |

Each VAL2 evaluation uses the same 34,103 patches,
with 24,471 benign and 9,632 tumour samples.

## Important limitations

- This is a constrained binary reimplementation, not an exact
  reproduction of the original paper.
- VAL1 was used for initial training because the original training
  patches and model weights were not released.
- Original paper training used a different cohort and class setup.
- SICAPv2 and Singapore fine-tuning datasets differ from VAL2
  in sampling, domain characteristics, and class balance.
- Negative transfer refers to deterioration at the fixed tumour
  probability threshold of 0.5.
- The relative roles of calibration shift and representation
  degradation have not been established.
- Model binaries are not included in this Git checkpoint.

See `README_15th_EPOCH.md` for the experimental comparison
and interpretation.
