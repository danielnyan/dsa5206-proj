# DSA5206 â€” Additional 15th-Epoch Fine-Tuning Experiments

**Scope:** Constrained binary NASNetLarge prostate histopathology reimplementation; 14-epoch local and Vanda baselines, followed by two **separate** epoch-15 fine-tuning experiments using SICAPv2 or the Singapore prostate gland dataset. All four checkpoints were evaluated independently on the **same VAL2** cohort. This is **not an exact reproduction** of the original paper's training protocol or three-class task.

## 1\. External datasets: sources and preparation

### 1.1 SICAPv2

* **Dataset name:** SICAPv2 â€” prostate H\&E pathology patches with Gleason-pattern annotations.
* **Original public source:** [SICAPv2 on Mendeley Data](https://data.mendeley.com/datasets/9xxm58dvs3/2) (dataset identifier `9xxm58dvs3`; see also [version 1](https://data.mendeley.com/datasets/9xxm58dvs3/1)).
* **Associated publication:** Silva-RodrÃ­guez et al., *Going deeper through the Gleason scoring scale: An automatic end-to-end system for histology prostate grading and cribriform pattern detection*, *Computer Methods and Programs in Biomedicine* 195 (2020), 105637.
* **Original task:** Multiple Gleason patterns / grading and localization, not the same as this project's binary benign-versus-tumour task.
* **Our epoch-15 preparation:** A binary-labelled, selected and preprocessed experimental subset of **9,959** patches: **3,773 benign** and **6,186 tumour**. The subset's counts should not be confused with the complete public dataset or original multiclass labels. Source images were extracted under the project's `data/sicapv2\_v2/extracted/SICAPv2/images` path; an experimental manifest and preprocessing checks were generated separately.
* **Protocol caveat:** This experiment used a **different field of view / magnification** from the original VAL1 task (recorded in the SICAP experiment plan as a wider-field secondary-domain deviation). Verify the exact preprocessing and optimizer state in its experiment artifacts before claiming a controlled head-to-head comparison with Singapore.

### 1.2 Singapore prostate gland classification dataset

* **Official source:** [Digital Pathology Dataset for Prostate Cancer Diagnosis â€” Zenodo record 7152243](https://zenodo.org/records/7152243), DOI [10.5281/zenodo.7152243](https://doi.org/10.5281/zenodo.7152243).
* **Creators:** Oner et al., including researchers at A\*STAR and Tan Tock Seng Hospital, Singapore.
* **Associated publication:** *An AI-assisted tool for efficient prostate cancer diagnosis in low-grade and low-volume cases*.
* **Original data:** Prostate H\&E whole-slide images and centred **512 Ã— 512 gland patches**. The official release includes gland segmentation and classification archives. Its published gland-classification table reports **10,652 labelled patches** (4,316 benign / 6,336 malignant); this is **not** the count used by the binary experimental manifest below.
* **Our epoch-15 preparation:** Selected `50\_100\_512` gland-patch configuration, combined original `train`, `valid` and `test` directories, applied the project's *experimental binary-label mapping / eligibility rules*, and constructed **11,020 eligible patches**: **4,337 benign** and **6,683 tumour**. The original extraction contained 12,777 rows; 1,757 were excluded by the experimental eligibility procedure. These counts differ from the published classification-table counts, so do **not** equate the custom labels with the paper's reported classification cohort without independently reconciling all annotation codes.
* **Original split sizes within our eligible binary manifest:** train **3,834**, validation **2,925**, test **4,261**; all were pooled for the extra epoch. **No independent Singapore test subset remains for post-fine-tuning evaluation** in this experiment. VAL2 was reserved for evaluation.
* **Vanda manifest:** `/scratch/e1536052/DSA5206/singapore\_external/manifests/singapore\_epoch15\_all.csv`
* **Cache:** `/scratch/e1536052/DSA5206/singapore\_external/cache\_epoch15/full11020/`

## 2\. Experimental design and data counts

The base model is a two-output **benign (0) / tumour (1)** NASNetLarge reimplementation. Because original paper training patches and trained weights were unavailable, released **VAL1** data were repurposed for training and internal validation. This is a constrained reproduction, with a potential limitation from patch-level splitting and unavailable patient/slide identifiers.

* **Base VAL1 training split:** 116,655 patches (**74,407 benign**, **42,248 tumour**), used in **each of 14 epochs**.
* **Separate VAL1 internal-validation split:** 29,164 patches (**18,602 benign**, **10,562 tumour**). This is **not** the held-out VAL2 test cohort and is not included in the 116,655 training count.
* **VAL2 evaluation cohort:** **34,103** patches (**24,471 benign**, **9,632 tumour**), identical across the four result columns.
* **Epoch 15 is continuation, not 15 epochs over pooled data:** Both external experiments start from the **Vanda 14-epoch** checkpoint, and each performs one additional epoch on its own external subset. Neither starts from the other epoch-15 checkpoint.

### 2.1 Unified dataset and evaluation comparison

|Field / VAL2 metric|Local 14 epochs (RTX 4080)|Vanda 14 epochs (A40)|Vanda + SICAPv2 epoch 15|Vanda + Singapore epoch 15|
|-|-:|-:|-:|-:|
|**Base training dataset**|VAL1|VAL1|VAL1|VAL1|
|Base training patches (benign / tumour)|**116,655** (74,407 / 42,248)|**116,655** (74,407 / 42,248)|**116,655** (74,407 / 42,248)|**116,655** (74,407 / 42,248)|
|Base training epochs|14|14|14|14|
|Additional epoch-15 dataset|None|None|SICAPv2 binary subset|Singapore gland-patch subset|
|**Additional epoch-15 patches (benign / tumour)**|â€”|â€”|**9,959** (3,773 / 6,186)|**11,020** (4,337 / 6,683)|
|**Cumulative source-patch counts (base + extra)**|116,655|116,655|126,614|127,675|
|**VAL1 internal validation (benign / tumour)**|29,164 (18,602 / 10,562)|29,164 (18,602 / 10,562)|29,164 (18,602 / 10,562)|29,164 (18,602 / 10,562)|
|**Evaluation cohort**|VAL2|VAL2|VAL2|VAL2|
|**VAL2 total (benign / tumour)**|**34,103** (24,471 / 9,632)|**34,103** (24,471 / 9,632)|**34,103** (24,471 / 9,632)|**34,103** (24,471 / 9,632)|
|**Accuracy**|**95.41%**|95.35%|70.09%|78.82%|
|**Precision (tumour)**|**93.33%**|93.21%|48.55%|57.33%|
|**Recall / sensitivity (tumour)**|90.18%|90.09%|**98.86%**|97.80%|
|**Specificity (benign)**|**97.46%**|97.42%|58.77%|71.35%|
|**F1 score (tumour)**|**91.73%**|91.62%|65.12%|72.29%|
|**AUROC**|0.9803|**0.9806**|0.9778|0.9729|
|True negatives (TN)|23,850|23,839|14,381|17,461|
|False positives (FP)|621|632|10,090|7,010|
|False negatives (FN)|946|955|110|212|
|True positives (TP)|8,686|8,677|9,522|9,420|
|**Interpretation at fixed 0.5 threshold**|Strong baseline|Strong baseline|Severe specificity / accuracy deterioration|Specificity / accuracy deterioration|

**Reading the counts:** The `base + extra` line adds source-patch counts across sequential training stages; it does **not** mean that the full combined dataset was used for all 15 epochs. Nor does the source-patch count prove cross-dataset sample uniqueness. The internal-validation set is shown separately and must not be counted as base training. All confusion-matrix values use rows = actual (`benign`, `tumour`) and columns = predicted (`benign`, `tumour`). The classification rule is **tumour when `p\_tumour > 0.5`** (ties remain benign). Percentages are rounded.

**Verification note:** Local and Vanda epoch-14 results and Singapore epoch-15 results are grounded in saved evaluation outputs. SICAPv2 evaluation metrics have been verified against the archived Vanda results at `experiments/epoch15/sicapv2/val2_results/metrics.json`.

### 2.2 Differences relative to the Vanda 14-epoch baseline

|Change on VAL2|SICAPv2 epoch 15|Singapore epoch 15|
|-|-:|-:|
|Accuracy|âˆ’25.26 percentage points|âˆ’16.52 percentage points|
|Tumour sensitivity|+8.77 percentage points|+7.71 percentage points|
|Benign specificity|âˆ’38.65 percentage points|âˆ’26.06 percentage points|
|Additional false positives|+9,458|+6,378|
|Fewer false negatives|845|743|
|AUROC|approximately âˆ’0.0028|approximately âˆ’0.0078|

## 3\. Likely explanations of negative transfer

**Observed result:** Both extra-epoch experiments increased tumour sensitivity but greatly reduced benign specificity and accuracy on VAL2 at the fixed `p\_tumour > 0.5` threshold. This is evidence of **negative transfer for this target evaluation and operating point**. It is **not**, by itself, proof of catastrophic forgetting or evidence that the networks can no longer distinguish cancer from benign tissue.

### 3.1 Fine-tuning class distribution differs from VAL2 â€” plausible driver

* **VAL2:** 9,632 / 34,103 = **28.24% tumour**.
* **SICAPv2 epoch-15 subset:** 6,186 / 9,959 = **62.12% tumour**.
* **Singapore epoch-15 subset:** 6,683 / 11,020 = **60.64% tumour**.

The extra-epoch datasets are tumour-majority, whereas VAL2 is benign-majority. With unweighted cross-entropy, the extra-epoch gradient is influenced disproportionately by tumour examples relative to the VAL2 distribution. A plausible consequence is a systematic increase in predicted tumour probabilities, yielding **more true positives as well as many more false positives**. Class balance alone, however, **does not establish causation**; it must be tested alongside calibration and image-domain differences.

### 3.2 Domain shift: visual scale, gland context, stain, and annotations

SICAPv2 contains Gleason-pattern-labelled prostate tissue patches with a **different spatial context / magnification** from the original VAL1 patches. Singapore provides **gland-centred** pathology patches with a distinct dataset construction and class-annotation scheme. Relevant variations include scanner and staining properties, field of view, gland/tissue morphology, the amount of surrounding stroma, and the precise benign/malignant label semantics. Normalizing image brightness and H\&E staining cannot undo differences in structural context or annotation conventions. Fine-tuning may therefore improve the external-domain objective while degrading performance on VAL2.

**Dataset-specific nuance:** SICAPv2 shows a more extreme specificity loss, whereas Singapore shows a somewhat larger AUROC reduction; these are not necessarily caused by exactly the same mechanism.

### 3.3 Updating a large portion of NASNetLarge â€” risk of representation drift

The restored Vanda model had approximately **208.8 million trainable parameters** at the fine-tuning boundary `normal\_conv\_1\_1`; the Singapore extra epoch used Adam with learning rate **1eâˆ’6**, batch size **100**, **111** updates, and retained the source optimizer state (iterations **1167 â†’ 1278**). A small dataset relative to the number of adjustable weights can move many feature representations toward the new domain, even over one epoch. Such changes could harm the original domain. A small learning rate is not a guarantee against negative transfer.

Avoid assuming optimizer-state parity between SICAPv2 and Singapore: the SICAPv2 experimental plan specified a **fresh Adam optimizer**, while Singapore explicitly resumed the original optimizer state. That difference is a **potential confounder**, not evidence for a single cause.

### 3.4 Probability-score shift versus discrimination loss

The AUROC changed only modestly relative to the very large specificity and accuracy reductions:

|Checkpoint|VAL2 AUROC|
|-|-:|
|Vanda epoch 14|**0.9806**|
|SICAPv2 epoch 15|**0.9778**|
|Singapore epoch 15|**0.9729**|

AUROC measures the ranking of tumour versus benign patches across possible thresholds. In contrast, the reported confusion matrices measure performance at the **fixed 0.5 threshold**. The pattern is **consistent with upward-shifted tumour probability scores / changed calibration**, since more benign and malignant patches crossed 0.5. The small but nonzero AUROC decreases also suggest **some genuine deterioration in ranking**. The available summary metrics cannot partition the effect precisely between calibration shift and altered feature discrimination.

### 3.5 Alternative possibilities and limitations

* **Small or selected external subsets:** Approximately 10â€“11k patches may not represent the full variability of benign VAL2 tissue, especially under a differing patch selection strategy.
* **Label harmonization:** Mapping SICAPv2 Gleason patterns and Singapore gland labels to a binary patch task can create semantic mismatch; verify exact mapping before drawing clinical conclusions.
* **No controlled ablation:** Dataset domain, class balance, patch selection, optimization protocol, and label semantics changed together. The current experiments cannot isolate an individual causal factor.
* **External validation constraints:** The Singapore train/valid/test subsets were all pooled for epoch 15, so a separate Singapore holdout cannot be claimed for that run. Results on VAL2 are independent of that fine-tuning data, but repeated investigation of VAL2 can influence future experimental decisions.

## 4\. Run artifacts and provenance

### Local, RTX 4080, epoch 14

* Local evaluation output: `runs/stage2i\_val2\_native\_epoch14/metrics.json`
* Model recorded in local evaluator log: `runs/stage2h\_production\_resume\_epoch11\_cuda\_malloc\_async/epoch14.keras`
* VAL2 evaluation: **34,103 patches**, batch size **1**.

### Vanda, A40, epoch 14

* Model: `/scratch/e1536052/DSA5206/vanda\_runs/nasnet\_production\_14epoch/epoch14.keras`
* VAL2 metrics: `/scratch/e1536052/DSA5206/vanda\_runs/val2\_epoch14\_b32\_1443006/metrics.json`
* Model SHA-256: `433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde`
* VAL2 evaluation: batch size **32**.

### SICAPv2, Vanda, epoch 15

* Original SICAPv2 evaluator (preserved): `/scratch/e1536052/DSA5206/dsa5206-proj/modern\_pca/vanda\_val2\_epoch15\_batch32.py`
* Original SICAPv2 VAL2 results (preserved): `/scratch/e1536052/DSA5206/vanda\_runs/val2\_epoch15\_sicap\_b32\_1443207/metrics.json`
* Fine-tuning plan observed locally: `reports/stage2j\_epoch15\_plan.json`
* **This experiment and its files are not overwritten by the Singapore experiment.**

### Singapore, Vanda, epoch 15

* Model: `/scratch/e1536052/DSA5206/singapore\_external/results/epoch15\_singapore/full11020/epoch15\_singapore.keras`
* Training summary: `/scratch/e1536052/DSA5206/singapore\_external/results/epoch15\_singapore/full11020/training\_summary.json`
* Model SHA-256: `5c1bf2f62c3fdb44a1dba239afa8718ab76a4d6b8b7be8b5195666ef3bc9dea8`
* New evaluator: `/scratch/e1536052/DSA5206/singapore\_external/scripts/evaluate\_val2\_singapore15.py`
* VAL2 result: `/scratch/e1536052/DSA5206/vanda\_runs/val2\_epoch15\_singapore\_1445854/metrics.json`
* Final status: **`VAL2 EVALUATION PASS`**, **34,103 / 34,103** images, batch size **32**.

## 5\. Overall conclusion

The locally trained and Vanda-trained 14-epoch NASNetLarge checkpoints produced **closely aligned VAL2 metrics** (95.41% and 95.35% accuracy). A single additional epoch using either SICAPv2 or Singapore patches then increased tumour sensitivity but strongly increased false-positive malignant predictions, reducing VAL2 accuracy and benign specificity. The observed negative transfer is plausibly related to **tumour-heavy fine-tuning distributions**, **domain and label mismatch**, **large-scale fine-tuning of network weights**, and **probability calibration/threshold shift**. The high remaining AUROC suggests that discrimination was **not completely lost**, but its modest decline warrants a careful ROC and score-distribution analysis. These are **hypotheses**, not mechanisms established by the current experiments.

## 6\. Dataset references

1. [SICAPv2 â€” Mendeley Data](https://data.mendeley.com/datasets/9xxm58dvs3/2).
2. Silva-RodrÃ­guez et al. (2020), *Going deeper through the Gleason scoring scale: An automatic end-to-end system for histology prostate grading and cribriform pattern detection*, *Computer Methods and Programs in Biomedicine*, 195, 105637.
3. [Oner et al. â€” Digital Pathology Dataset for Prostate Cancer Diagnosis](https://zenodo.org/records/7152243), Zenodo, DOI: [10.5281/zenodo.7152243](https://doi.org/10.5281/zenodo.7152243).
4. [An AI-assisted tool for efficient prostate cancer diagnosis in low-grade and low-volume cases](https://pmc.ncbi.nlm.nih.gov/articles/PMC9768677/).
