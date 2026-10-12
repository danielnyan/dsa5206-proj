# VAL2 External Evaluation: Presentation Brief

**Project:** DSA5206 prostate cancer pathology reproducibility study  
**Evaluation runs:** 10-11 October 2026
**Primary result:** On all 34,103 VAL2 patches, the naive/native model achieved 95.35% accuracy, P8 95.84%, routed ENS 95.85%, and pure Model 2 95.62%.
**Interpretation:** The ENS improvement is in the same direction as the paper's reported result, but this is a constrained binary reimplementation, not an exact reproduction.

## Executive Summary

- Evaluated the naive/native checkpoint, P8, the paper-inspired two-model ENS routing strategy, and pure Model 2 on the complete held-out VAL2 cohort. P8's aggregate values are transcribed from the user's supplied results; the other figures are present in local run artifacts.
- VAL2 contains 24,471 benign and 9,632 tumour patches. No VAL2 training, threshold tuning, or model selection was performed.
- Naive/native accuracy was **95.35%** (AUC **0.9806**); P8 accuracy **95.84%**; routed ENS accuracy **95.85%** (AUC **0.9808**); and pure Model 2 accuracy **95.62%** (AUC **0.9816**).
- ENS used Model 2 only for **1,069 / 34,103 patches (3.13%)** whose native tumour probability was in the inclusive 0.2-0.8 uncertainty band. It replaced the native score for those patches; it did not average model outputs.
- Sensitivity increased from **90.09% to 91.38%**. False negatives decreased by 125, while false positives decreased by 47. Specificity also increased slightly, from 97.42% to 97.61%.
- The ENS full evaluator runtime was **17,894.6 seconds (4 h 58 min 15 sec)**; the pure Model 2 full run took **29,336.8 seconds (8 h 8 min 57 sec)** on its evaluator timer. Neither figure includes queue wait.
- Compared with the publication's reported VAL2 accuracies for native, P8, and ENS, our results are lower by **1.35 pp**, **1.46 pp**, and **1.15 pp**, respectively. Pure Model 2 has no corresponding paper VAL2 row in the supplied comparison.

## Study Context and Work Completed

The evaluation follows the VAL2 design of Tolkach et al., *High-accuracy prostate cancer pathology using deep learning* (Nature Machine Intelligence, 2020; [DOI: 10.1038/s42256-020-0200-7](https://doi.org/10.1038/s42256-020-0200-7)). The project prepared a checksum-audited dataset, retained VAL1 and VAL2 as separate cohorts, trained constrained binary models using VAL1, and then used VAL2 once as a final external evaluation.

The work relevant to this result included:

1. Downloaded and verified the authors' VAL2 archives from Zenodo record 3825933 using the published MD5 checksums.
2. Built a frozen, deterministic VAL2 manifest with sample identity, ground-truth class, image path, image SHA-256, file size, dimensions, and mode. The manifest contains 34,103 images and has SHA-256 `7b95ac02798e5971fc2722d3c37e8e7613bc1d020dee9a1a5d49ac939cf746d9`.
3. Kept VAL2 outside the training and model-selection workflow. Both checkpoints were trained using the VAL1 development data; the evaluation runner rejects a wrong manifest, checkpoint, archive, or stain reference.
4. Implemented a separate inference-only VAL2 runner. It checks archive MD5s, the manifest fingerprint, each image's SHA-256, model checkpoint hashes, and the stain-reference hash before writing results.
5. Ran a 1,024-patch smoke test (512 per class) to check data access, preprocessing, model loading, scoring, routing, and report generation.
6. Ran the final full-cohort evaluation and saved per-patch predictions, aggregate metrics, and a completion receipt with artifact checksums.

The primary evidence is in [the full-run metrics](evaluation_ens_val2_full_20261010_163459/val2_ens_metrics.json), [full-run predictions](evaluation_ens_val2_full_20261010_163459/val2_ens_predictions.csv), and [completion receipt](evaluation_ens_val2_full_20261010_163459/run_complete.json). The evaluator implementation is in [ens_val2.py](../ens_model_training/ens_val2.py); dataset counts and manifest fingerprint are recorded in [the manifest summary](../manifests/manifest_summary.md).

## Dataset and Split

| Cohort | Benign | Tumour | Total | Role |
|---|---:|---:|---:|---|
| VAL1 | 93,009 | 52,810 | 145,819 | Training and internal validation only |
| VAL2 | 24,471 | 9,632 | 34,103 | Held-out final evaluation |

VAL2 class proportions are approximately **71.76% benign** and **28.24% tumour**. The class imbalance makes accuracy alone incomplete, so sensitivity, specificity, precision, F1, ROC-AUC, and confusion counts are also reported. All 34,103 VAL2 patches are included in the full-run metrics; the smoke subset is not used as the headline result.

## Models and Evaluation Method

### Checkpoint provenance

| Model | Role | Input | Output and provenance |
|---|---|---:|---|
| Native Model 1 (naive model) | First-stage classifier and native-only baseline | 350 x 350 | Two-class NASNetLarge; epoch 14; checkpoint `naive model_epoch14.keras`; SHA-256 `433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde` |
| Model 2 | Uncertainty-stage classifier | 500 x 500 | Two-class NASNetLarge; epoch 14; checkpoint SHA-256 `49305a192b3dfbdad7c9cea641f4195ae096389df0655863022c01f0d58a4680` |

P8 is included as a separately evaluated model in the comparison below. Its checkpoint identity, preprocessing details, and raw VAL2 output artifact were not found in the local report files, so those implementation details cannot be audited from the current workspace.

The models use ImageNet-initialized NASNetLarge backbones with a Flatten layer, Dense(256, ReLU), and a two-unit softmax output ordered benign/tumour. The training pipeline used brightness standardization and Macenko stain normalization with a fixed stain reference, flips, and a staged 14-epoch fine-tuning schedule. Model 2 was trained on VAL1 using the Model 1 recipe adapted to 500-pixel inputs. Its VAL1 internal-validation accuracy at epoch 14 was 97.11%; this is model-development context, not the external VAL2 result.

This is not the authors' original three-class experiment. The original study used benign glandular, benign non-glandular, and tumour labels, while the released validation patches available here have binary benign/tumour labels. The project therefore used a constrained two-class reimplementation, with the released VAL1 data repurposed for training and internal validation. The exact original training data and authenticated original three-output checkpoint are unavailable. See [the reimplementation limitations](../REIMPLEMENTATION.md) and [Model 2 training/evaluation notes](../ens_model_training/README.md).

### Inference and ENS routing

1. Read each image from the authenticated VAL2 archive and verify its SHA-256 against the frozen manifest.
2. Apply the project's legacy-compatible RGB decode, resize, brightness standardization, and Macenko stain normalization using the fixed reference image.
3. Score every patch with native Model 1 and obtain its tumour probability, `p_native`.
4. If `p_native` is outside `[0.2, 0.8]`, retain the native score. If it is inside the inclusive band, run Model 2 on that patch and use its tumour probability instead.
5. Classify as tumour when the final probability is **greater than 0.5**; otherwise classify as benign. Exact 0.5 ties are benign.

There is no probability averaging, VAL2 threshold search, test-time augmentation, or validation-time model selection. ROC-AUC uses the final continuous scores. The confusion matrix is ordered `[[TN, FP], [FN, TP]]` with benign first and tumour second.

## Evaluation Duration and Execution

| Run | Samples | Result | Recorded elapsed time |
|---|---:|---|---:|
| ENS smoke test | 1,024 (512/class) | Pipeline check only; not representative headline performance | 709.4 sec (11 min 49 sec) |
| Model 2-only smoke test | 1,024 (512/class) | Standalone-path check only; not representative headline performance | 953.1 sec (15 min 53 sec) |
| Naive/native Model 1 | 34,103 | Scored as the native baseline within the full ENS evaluation; no standalone timer | Not separately recorded; shared run took 17,894.6 sec (4 h 58 min 15 sec) |
| Routed ENS | 34,103 | Final external evaluation with Model 2 invoked on 1,069 patches | 17,894.6 sec (4 h 58 min 15 sec) |
| Pure Model 2 | 34,103 | Model 2 scored every patch | 29,336.8 sec (8 h 8 min 57 sec) |
| P8 | 34,103 | Aggregate metrics supplied by user; run timer not available locally | Not recorded |

The ENS full run completed at **2026-10-10 13:33:48 UTC** (21:33:48 Singapore time); the Model 2-only full run completed at **2026-10-11 13:23:41 UTC** (21:23:41 Singapore time). Naive/native has no standalone full-cohort runtime because its predictions were generated as part of the ENS run; do not present the shared 4 h 58 min as native-only inference time. The evaluator timer begins inside the Python evaluation function after TensorFlow has been imported. It includes manifest and source checks, model loading, preprocessing, inference, metric calculation, and prediction CSV writing, but excludes metrics JSON and completion-receipt writing, PBS queue wait, and Python/TensorFlow process startup before the timer. The ENS full run averaged about **1.91 patches/second** over the evaluator's measured interval; this is end-to-end evaluator throughput, not isolated GPU inference throughput. The Model 2-only full-run elapsed time is read from its metrics JSON and uses the same evaluator's elapsed-time field. No comparable duration evidence is available for P8.

The PBS script requested one GPU, eight CPUs, and a 12-hour walltime on NUS Vanda. The local metrics record TensorFlow 2.21.0 and Keras 3.15.1. The actual GPU model, PBS job ID, queue delay, and full scheduler log are not preserved alongside the local result artifacts. Do not describe 12 hours as the measured duration; it is the requested walltime limit.

## Full-Cohort Results

| Metric | Native Model 1 | ENS | Change (ENS - native) |
|---|---:|---:|---:|
| Accuracy | 95.35% | **95.85%** | **+0.50 pp** |
| Precision | 93.21% | 93.77% | +0.56 pp |
| Sensitivity / tumour recall | 90.09% | **91.38%** | **+1.30 pp** |
| Specificity / benign recall | 97.42% | 97.61% | +0.19 pp |
| F1 score | 91.62% | **92.56%** | +0.94 pp |
| ROC-AUC | 0.98063 | 0.98075 | +0.00012 |

Metrics are computed over the complete 34,103-patch cohort. `pp` means percentage points. AUC values are rounded for display; full precision is in the JSON artifact.

### Confusion Counts

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Native | 23,839 | 632 | 955 | 8,677 |
| ENS | **23,886** | **585** | **830** | **8,802** |

The routed result corrected 125 additional tumour patches (FN fell from 955 to 830) and 47 additional benign patches (FP fell from 632 to 585), relative to native. These are net confusion-count changes, not a claim that exactly these same number of individual errors were corrected without any error reversals.

### Routing Efficiency

- Model 2 invoked: **1,069 / 34,103 = 3.13%** of patches.
- Model 1 alone handled the remaining **33,034 (96.87%)**.
- Model 2 trigger fraction: 3.13%, recorded by the full-run metrics.
- This suggests that, under this fixed threshold rule and on this cohort, the second model was needed for a small uncertainty subset. No timing claim about the counterfactual cost of running both models on every patch was measured in this evaluation.

### Results Across All Full VAL2 Evaluations

The naive/native and ENS values below come from the same full-run artifact: native is the first-stage checkpoint scored by itself, while ENS applies the fixed uncertainty-routing rule. Pure Model 2 is a separate full-cohort run. P8 values were supplied for this report but could not be checked against a local metrics file.

| Model / evaluation | Precision | Tumour recall / sensitivity | F1 | Overall accuracy | Benign accuracy / specificity | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Naive / native Model 1 | 0.932 | 0.901 | 0.916 | 95.35% | 97.42% | 0.98063 |
| P8 | 0.942 | 0.909 | 0.925 | 95.84% | 97.80% | Not available |
| Routed ENS | 0.938 | 0.914 | 0.926 | **95.85%** | 97.61% | 0.98075 |
| Pure Model 2 | 0.919 | **0.926** | 0.923 | 95.62% | 96.79% | **0.98163** |

**Cross-model findings:**

- Pure Model 2 has the highest tumour recall: **92.64%**, detecting 8,923 of 9,632 tumour patches and missing 709. It produces 786 false positives.
- ENS has the highest overall accuracy among these four results (**95.85%**) and fewer total errors than pure Model 2: 1,415 versus 1,495. Relative to Model 2, ENS has 121 more false negatives but 201 fewer false positives.
- P8 and ENS have nearly identical displayed overall accuracy (95.84% and 95.85%). P8 has higher precision and benign accuracy; ENS has higher tumour recall and F1 in the supplied figures. Since P8's exact counts and unrounded metrics are unavailable here, avoid over-interpreting this small ranking difference.
- Relative to naive/native, P8, ENS, and pure Model 2 increase tumour recall by about 0.82, 1.30, and 2.55 percentage points, respectively. Their trade-offs differ: Model 2's higher tumour recall comes with lower precision and benign accuracy.
- The routed ENS invoked Model 2 for only 1,069 patches (3.13%), whereas pure Model 2 evaluated all 34,103 patches. These are different evaluation modes, and their recorded runtimes should not be treated as a controlled speed benchmark without matched hardware and scheduler records.

Exact confusion counts are available for naive/native, ENS, and pure Model 2. P8's per-class counts were not supplied, so they are intentionally omitted rather than inferred from rounded percentages.

## Smoke-Test Findings

The deterministic smoke subset contained 512 benign and 512 tumour images selected from the manifest order per class. Native accuracy was 93.75%; ENS accuracy was 94.53%; Model 2 was invoked for 37 patches (3.61%). The smoke result established that the pipeline executed end-to-end. Because this was a small, manifest-ordered subset rather than the full cohort, these metrics should not be interpreted as an estimate of full VAL2 performance or used as the final result.

The separate Model 2-only smoke run evaluated the same 1,024 sample identities, invoked Model 2 on
every patch, and achieved 94.04% accuracy in 953.1 seconds. It validated the standalone inference
path before the full Model 2 evaluation; it is not a headline external-validation result.

## Paper Versus This Evaluation

| Model | Source | Precision | Tumour recall | F1 | Overall accuracy | Benign accuracy |
|---|---|---:|---:|---:|---:|---:|
| Native / naive | Paper | 0.927 | 0.957 | 0.942 | 96.7% | 97.1% |
| Native / naive | Ours | 0.932 | 0.901 | 0.916 | 95.35% | 97.42% |
| Native / naive | Ours minus paper | +0.005 | -0.056 | -0.026 | -1.35 pp | +0.32 pp |
| P8 | Paper | 0.944 | 0.962 | 0.953 | 97.3% | 97.8% |
| P8 | Ours | 0.942 | 0.909 | 0.925 | 95.84% | 97.80% |
| P8 | Ours minus paper | -0.002 | -0.053 | -0.028 | -1.46 pp | 0.00 pp |
| ENS | Paper | 0.932 | 0.962 | 0.947 | 97.0% | 97.3% |
| ENS | Ours | 0.938 | 0.914 | 0.926 | 95.85% | 97.61% |
| ENS | Ours minus paper | +0.006 | -0.048 | -0.021 | -1.15 pp | +0.31 pp |
| Pure Model 2 | Paper | No corresponding row supplied | No corresponding row supplied | No corresponding row supplied | No corresponding row supplied | No corresponding row supplied |
| Pure Model 2 | Ours | 0.919 | 0.926 | 0.923 | 95.62% | 96.79% |

The paper comparison does not include a pure Model 2 row. Its VAL2 result is therefore reported as an additional ablation/baseline, not as a direct reproduction of a published paper result.

**Did the result replicate?** The answer depends on what “replicate” means:

- **Qualitative method/result pattern: broadly consistent.** ENS improved accuracy over the native model, and most of the sensitivity gain came from tumour recall (+1.30 pp versus +0.19 pp specificity). This is consistent with the paper's description of the benefit as being mainly in tumour sensitivity.
- **Published headline values: not numerically reproduced.** Native, P8, and ENS accuracy are all below the paper's reported VAL2 values. Pure Model 2 has no corresponding paper value in the supplied comparison.
- **Exact experimental reproduction: no.** This project does not have the original three-class training data or authenticated original checkpoint. It uses a binary task, VAL1-trained constrained models, and a modern reimplementation. The paper comparison is therefore descriptive, not a controlled like-for-like comparison.

The paper-level comparison figures above are the values recorded in the project notes and supplied results; consult the publication itself for exact table/figure attribution before putting those numbers on a slide. The available local evidence does not establish why the absolute accuracy differs, so differences in training data/task, checkpoint, and implementation are caveats, not proven causal explanations.

## Limitations and Claims to Avoid

- Do not call this an exact reproduction of the paper or the authors' original model.
- Do not imply patient-level or slide-level independence. The available patch metadata does not provide reliable patient/WSI identifiers, and patch-level leakage is possible in the VAL1 development split.
- Do not imply that the smoke test is the final result; use only the full-cohort metrics for the principal finding.
- Do not describe the 12-hour PBS walltime request as actual evaluation duration.
- Do not infer that the measured 3.13% routing rate proves lower total compute or clinical deployment suitability; no paired cost, latency, calibration, or clinical utility study was done here.
- The evaluation is patch-level and binary. It is not a patient-level diagnostic study and does not evaluate Gleason grading.
- Accuracy, sensitivity, and specificity are threshold-dependent; AUC measures ranking across thresholds. The threshold and routing band were fixed before the VAL2 run in the evaluator.

## Suggested Slide Sequence

1. **Question and study design:** Can the paper-inspired ENS rule improve a VAL1-trained constrained model on untouched VAL2?
2. **Data and leakage safeguards:** Zenodo source, VAL1/VAL2 separation, 34,103 VAL2 patches, checksums, class balance.
3. **Models and method:** naive/native, P8, pure Model 2, and routed ENS; fixed `[0.2, 0.8]` routing band and `p > 0.5` decision for the ENS evaluation.
4. **Execution and reproducibility:** smoke test, ENS full run, Model 2-only full run, checksum validation, evaluator durations, and output artifacts.
5. **Main metrics:** four-model comparison with accuracy, sensitivity, specificity, F1, and AUC where available.
6. **Trade-offs:** highest tumour recall for pure Model 2, highest overall accuracy for ENS, plus P8/ENS near-tie and routing frequency.
7. **Paper comparison and replication verdict:** directionally consistent ENS gain, lower absolute accuracy for the comparable models, and no direct paper baseline for pure Model 2.
8. **Limitations and next steps:** original checkpoint/data, patient-level grouping, hardware/job log capture, confidence intervals or additional external cohorts.

## Evidence and Reproducibility Artifacts

- [Full-run metrics JSON](evaluation_ens_val2_full_20261010_163459/val2_ens_metrics.json)
- [Full-run per-patch predictions CSV](evaluation_ens_val2_full_20261010_163459/val2_ens_predictions.csv)
- [Full-run completion receipt and output hashes](evaluation_ens_val2_full_20261010_163459/run_complete.json)
- [Full Model 2-only metrics](evaluation_ens_val2_full_20261010_163459_model2_only_val2_20261011_131430/val2_model2_metrics.json)
- [Full Model 2-only per-patch predictions](evaluation_ens_val2_full_20261010_163459_model2_only_val2_20261011_131430/val2_model2_predictions.csv)
- [Full Model 2-only completion receipt](evaluation_ens_val2_full_20261010_163459_model2_only_val2_20261011_131430/run_complete.json)
- [Smoke-run metrics JSON](evaluation_ens_val2_full_20261010_163459_ens_val2_smoke_20261010_161931/val2_ens_metrics.json)
- [Model 2-only smoke-run metrics JSON](evaluation_ens_val2_full_20261010_163459_model2_only_val2_20261011_124016/val2_model2_metrics.json)
- [VAL2 inference implementation](../ens_model_training/ens_val2.py)
- [Vanda PBS job configuration](../ens_model_training/jobs/ens_val2_vanda.pbs)
- [VAL2 manifest](../manifests/VAL2_manifest.csv)
- [Dataset counts and manifest fingerprints](../manifests/manifest_summary.md)
- [Dataset source and official archive checksums](../DATASET.md)
- [Binary reimplementation scope and limitations](../REIMPLEMENTATION.md)
- [Model 2 recipe, training results, and evaluation instructions](../ens_model_training/README.md)

## Details to Confirm Before Final Slides

The results and method are supported by the local artifacts. For a fully documented presentation, add or confirm:

- The Vanda PBS job ID, assigned GPU model, and scheduler log, if available; these would let the slide state the observed hardware and queue/runtime breakdown rather than only the requested resources and evaluator timer.
- The original paper's exact table/figure reference for the 96.7% native and 97.0% ENS VAL2 values, and whether the paper's reported protocol uses exactly the same threshold conventions.
- The P8 full-run metrics JSON/prediction file, checkpoint hash, and run log. The P8 aggregate figures in this document are currently transcribed from the supplied message, not independently verified from workspace artifacts.
- Confirm whether “naive model” should be labelled “Native Model 1” on the slides. This report treats it as the native-only run of `naive model_epoch14.keras`, matching the checkpoint hash and evaluation configuration.
- Whether presentation policy requires confidence intervals. The current report contains point estimates only; no bootstrap or patient-clustered uncertainty interval was computed.
- Whether the audience expects a dedicated comparison of the paper's training data and model outputs. That would require the paper's exact cohort/checkpoint details and should remain clearly separated from the constrained binary experiment here.
