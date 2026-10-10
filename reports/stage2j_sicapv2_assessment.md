# Stage 2J — SICAPv2 one-epoch external post-training feasibility

**CONDITIONAL GO** for one predefined SECONDARY external fine-tuning experiment, conditional on accepting the documented 10× magnification/domain deviation. This is **not** reproduction of Tolkach's original-model → VAL1 post-training procedure. No improvement on VAL2 is promised.

## Acquisition and provenance

Official [Version 2, DOI 10.17632/9xxm58dvs3.2](https://data.mendeley.com/datasets/9xxm58dvs3/2), published 22 October 2020, CC BY 4.0. Downloaded outside Git to `/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/data/sicapv2_v2/SICAPv2.zip`.

- Exact download: https://data.mendeley.com/public-files/datasets/9xxm58dvs3/files/6ab087a7-ca89-47ac-9698-f6546bb50f98/file_downloaded
- Archive bytes: 2,159,052,394; SHA256: `a28aa6a0e3831217bf31575cecfc1846607ace7ff25a4392f07052bbc7355916` (matches official API record).
- 37,589 files; 2,161,978,547 uncompressed bytes.
- 18,783 JPEG images, 18,783 PNG masks, 21 XLSX files, readme.txt and Thumbs.db. No higher-resolution WSI files.
- Extraction: `/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/data/sicapv2_v2/extracted/SICAPv2`. Every extracted file hash is recorded in `/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/data/sicapv2_v2/audit/archive_members.csv`; image/mask details in `images.csv` beside it.
- Image properties: `{'{"format": "JPEG", "height": 512, "mode": "RGB", "width": 512}': 18783}`. Mask properties: `{'{"format": "PNG", "height": 512, "mode": "L", "width": 512}': 18783}`.
- 155 slide identifiers and 95 patient identifiers are explicitly provided by `wsi_labels.xlsx`, with primary/secondary global Gleason grades.
- Bundled readme has a stale “Version 1: 2020-05” header. Official version, archive hash and observed contents take precedence; this inconsistency is retained in the evidence.

## Labels and partition selection

Use **only `partition/Test/Train.xlsx`**, the main official training partition (the parent directory name “Test” denotes the held-out partition scheme). `NC→benign0`; `G3/G4/G5→tumour1`. `G4C` is ancillary cribriform information, not a fifth mutually exclusive class. All primary labels are one-hot and agree across supplied classification tables.

| Candidate | Samples | Benign | Tumour | Patients | Slides |
|---|---:|---:|---:|---:|---:|
| Recommended official train | 9959 | 3773 | 6186 | 74 | 124 |
| All official train + test (not selected) | 12081 | 4417 | 7664 | 95 | 155 |

Exclude **8,824** released images from this experiment: **2,122 official test** plus **6,702 not listed in the main classification train/test tables**. There are **0 ambiguous official primary labels** and **0 preprocessing/corruption exclusions** from the selected 9,959. Unlisted masks are not converted into new training labels. Do not enlarge training with the held-out test. Main train/test patient overlap is zero; all four supplied internal CV folds are also patient-disjoint and partition the main training set. Exact fold class/patient/slide counts are in the JSON.

Tumour prevalence in the selected set is 62.11%. Use no class weighting or resampling. Keep mixed morphology patches with valid official primary labels; the patch classification task does not require every pixel to share that label.

## Mask interpretation and quality

**Actual mask values are exactly0/3/4/5, disagreeing with the readme's0/1/2/3 legend.** Literal global histogram: `{'mask_other': 0, 'mask_value_0': 4220631721, 'mask_value_1': 0, 'mask_value_2': 0, 'mask_value_3': 190847349, 'mask_value_4': 428337950, 'mask_value_5': 84033732}`. Majority-nonzero versus official labels: `{'NC': {'0': 4417}, 'G3': {'3': 2220, '4': 2}, 'G4': {'4': 4490, '3': 4}, 'G5': {'5': 948}, 'unlisted': {'5': 90, '4': 185, '3': 120, '0': 6307}}`. The observed association supports0=NC/unannotated/background and3/4/5=G3/G4/G5. Six official G3/G4 labels disagree with mask majority grade (four G4→majority3; two G3→majority4), but all remain **tumour** under the proposed binary mapping; retain official labels. Do not silently relabel or threshold masks. Zero does not separately encode tissue versus background. Multiple nonzero values show mixed annotations but cannot independently certify biological mixtures or distinguish boundary effects. Numeric mixtures: 8,055 zero/nonzero and 6,638 multiple-nonzero-value patches. Of6,702 unlisted images,395 have nonzero annotations and6,307 have all-zero masks; neither group is added to training.

All 18,783 image/mask pairs decode and match dimensions; **0 corrupt/unreadable pairs**, **0 raw-byte duplicate groups**, **0 decoded-RGB duplicate groups**. Exact JPEG hash overlaps: **VAL1=0; VAL2=0**, compared against authenticated existing manifests only. No VAL2 images opened. Hash checks do not rule out re-encoded/near duplicates or establish cross-dataset patient identity. Within-slide 50% patch overlap creates expected spatial correlation; patient partitions are preserved.

QC: {'all_white_images': 0, 'constant_images': 0, 'max_bright_fraction': 0.7818374633789062, 'min_rgb_std': 18.164771656675907, 'mean_rgb_ranges': {'r': [105.03046035766602, 250.91845321655273], 'g': [68.22867202758789, 238.74380111694336], 'b': [126.70899963378906, 245.9622573852539]}, 'meaning': 'Brightness >=240 in all channels and RGB std are descriptive QC only; no exclusion thresholds.', 'normalized_constant_images': 0, 'normalized_std_range': [0.16231784224510193, 0.32279691100120544], 'normalized_mean_rgb_ranges': [[0.5145388245582581, 0.9440447688102722], [0.35719436407089233, 0.8846815228462219], [0.5575470924377441, 0.9237765669822693]]}. These are descriptive values, not exclusion thresholds.

## Scale and preprocessing

The released readme documents 512×512 patches at10× with50% overlap. The [authors' paper](https://arxiv.org/html/2105.10490v1) describes40× source scanning and10× patch generation. No calibrated MPP or physical field of view is established by the released JPEG metadata, and this archive contains no higher-resolution WSI source. The primary repository pipeline uses600px at40×, resized to350 (`MAIN.py:12–14`, `wsi_process.py:35–53`). Under comparable calibration, SICAP's whole-patch field is approximately **3.41× wider /11.65× the area**. These are nominal ratios, not measured micrometre values.

The proposed whole512→350 resize therefore retains a substantially different morphology scale. **Interpolation cannot restore absent optical detail.** A crop or alternative resize would define another protocol and could invalidate whole-patch labels; neither is proposed.

**9,959/9,959 selected images passed** the unchanged PIL RGB → LANCZOS350 → frozen brightness standardizer → StainTools Macenko → float32/255 path; **0 failures (0%)**. Same reference SHA `1d31d575c21dcbcf172957b0783c37fcb07872d3129b3a7b3e86ede25ea0a1c1`. Full measured preprocessing audit: 20.71min. No cache was built. [Raw/normalized contact sheet](stage2j_preprocessing_contact_sheet.png): eight deterministic samples, two per official grade. Stronger contrast and slight background speckling are visible; no gross structural loss seen at preview scale. This is not exhaustive visual or histopathologist validation.

## Frozen epoch15 and feasibility

Start `runs/stage2h_production_resume_epoch11_cuda_malloc_async/epoch14.keras` (SHA `5cecd53138d74cffc7313521a1c78544096d7e872440b496631a5666534e1e61`), exactly one extra epoch; final boundary `normal_conv_1_1`; frozen/inference BN; fresh Adam LR1e-6, beta1=.9, beta2=.999, epsilon1e-7; no decay/clipping/AMSGrad; unweighted sparse categorical cross-entropy; physical4/effective100; same stateless horizontal/vertical flips atp=.5, seed42; float32; TF32off; cuda_malloc_async. Deterministic external manifest order and epoch15 flip key are frozen. Every selected image appears once. No SICAP test validation, checkpoint selection, VAL2-based adaptation, or14-epoch schedule restart.

**100 optimizer updates**: 99 groups of100 then59; 2,490 physical batches (last3). No updates were executed during this assessment. GPU visibility and a read-only4-image forward/backward probe passed with epoch14 weights, Adam slots and accumulation buffers allocated. Peak **9.972GB**, current **4.186GB**. Weights, BN, Adam and buffers unchanged. Eager peak is close to the current available GPU budget; this is not a full-epoch guarantee. Prior same-boundary compiled Stage2H evidence peaked at5.857GB and processed13.392images/s. Keep other GPU workloads closed.

Serial preprocessing plus historical compute estimate: 32.5min; allow **35–55min** including tracing, I/O and checkpoint/export. Hardware contention may increase this. No large cache needed; optional uint8 tensors alone would be 3.660GB. Reserve approximately5GB for the secondary model/Adam checkpoint/export/logs. Available data-volume space: 1061.6GB.

The secondary trainer saves `epoch_15_resume/`, `epoch15.keras`, standalone `history.json`, `metadata.json`, `run.log`, and `run_complete.json` into a fresh directory. Primary epoch1–14 history/checkpoints remain unchanged. It has no mid-epoch resume feature; an interrupted epoch is incomplete and must not be evaluated. Never rerun in response to VAL2 results.

## Comparison frozen before training

A = Stage2I native epoch14 VAL2; B = one SICAP epoch15 then VAL2. Compare accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC and TN/FP/FN/TP; save each image's unrounded tumour-probability shift B−A and threshold crossings. Fixed rule p_tumour>0.5, ties benign; batch1 GPU inference, no augmentation/TTA/calibration. Do not inspect VAL2 errors to select samples, LR, threshold, epochs, preprocessing or repeat/select runs. Report worse outcomes as worse.

User-provided approximate descriptive paper values only: native .967/.927/.957/.942/.9918; paper post-VAL1 .974/.951/.955/.953/.9939 (accuracy/precision/recall/F1/AUC). Our SICAP sequence is scientifically different. These numbers are not targets. Stage2I outcome artifacts were not inspected during this assessment.

## Validation and boundaries

- Focused tests: 24 passed in 7.32s.
- Full tests: 303 passed in 60.83s (0:01:00).
- `git diff --check`: PASS; new-file trailing whitespace check: PASS.
- Primary frozen source hashes and checkpoint/history hashes: unchanged.
- Training started: **NO**. New VAL2 evaluation: **NO**. VAL2 image access: **NO**. Commit/push: **NO**.
- Added code: `modern_pca/sicapv2.py`, `modern_pca/train_sicapv2.py`, `modern_pca/evaluate_sicapv2.py`, `tests/test_sicapv2.py`, `tests/test_sicapv2_training.py`, `tools/inspect_sicapv2.py`, `tools/audit_sicapv2.py`, `tools/check_sicapv2_preprocessing.py`, `tools/verify_stage2j.py`, `tools/report_stage2j.py`. Reports/evidence use only `reports/stage2j_*`; dataset stays outside Git.

## Proposed commands — NOT EXECUTED

Training, only if the10× secondary deviation is accepted:

```bash
cd '/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/GitHub/dsa5206-proj'
TF_GPU_ALLOCATOR=cuda_malloc_async /home/soonh/miniconda3/envs/dsa5206-reimplementation/bin/python -B -u -m modern_pca.train_sicapv2 \
  --plan reports/stage2j_epoch15_plan.json \
  --model runs/stage2h_production_resume_epoch11_cuda_malloc_async/epoch14.keras \
  --manifest '/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/data/sicapv2_v2/audit/selected_train.csv' \
  --images '/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/data/sicapv2_v2/extracted/SICAPv2/images' \
  --output runs/stage2j_sicapv2_epoch15 \
  --train-one-epoch --accept-magnification-deviation
```

After successful epoch15 and an authenticated completed Stage2I baseline, one fixed secondary VAL2 evaluation:

```bash
cd '/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/GitHub/dsa5206-proj'
env -u TF_GPU_ALLOCATOR /home/soonh/miniconda3/envs/dsa5206-reimplementation/bin/python -B -u -m modern_pca.evaluate_sicapv2 \
  --plan reports/stage2j_epoch15_plan.json \
  --run runs/stage2j_sicapv2_epoch15 \
  --baseline runs/stage2i_val2_native_epoch14 \
  --manifest manifests/VAL2_manifest.csv \
  --extracted '/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/data/zenodo/_3825933/extracted' \
  --output runs/stage2j_val2_epoch15 --final-evaluation
```

The evaluator verifies the secondary completion marker/model hash and baseline identity, uses the same frozen inference/metric helpers, and writes separate outputs. It does not relax Stage2I's epoch14-only contract. Source, software or plan changes require revalidation; these commands fail closed on drift.
