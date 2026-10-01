# Stage 2B manifest summary

Source: Zenodo 3825933. Validation: **PASS**.

Source images were not modified. No resizing, resampling, renaming, moving, random splitting or cohort merging occurred.

| Cohort | Benign | Tumour | Total | Cohort CSV SHA-256 |
|---|---:|---:|---:|---|
| VAL1 | 93009 | 52810 | 145819 | c483f827985c241106942766062ad58b23c7ad86e6853ade51bb045bb5feff77 |
| VAL2 | 24471 | 9632 | 34103 | 7b95ac02798e5971fc2722d3c37e8e7613bc1d020dee9a1a5d49ac939cf746d9 |

Generation UTC: 2026-10-01T05:49:29.330780+00:00

Stage 2A references: reports\dataset_verification.json and reports\dataset_verification.md

relative_path = class_name/archive-relative-path; resolve by stripping class_name/ and joining the corresponding source_root; sample_id = cohort:relative_path.

Zero-based contiguous indices independently assigned after sorting each CSV; join on sample_id.

All 1,837 native dimension variants are retained. Native modes are listed below.

## VAL1

Dimensions (width x height): {"609x612": 322, "610x609": 134, "610x610": 147, "610x611": 352, "610x612": 144864}
Modes: {"RGB": 145819}

benign source root: D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\data\zenodo\_3825933\extracted\val_dataset_1_norm
Dimensions: {"609x612": 209, "610x609": 70, "610x610": 86, "610x611": 229, "610x612": 92415}
Modes: {"RGB": 93009}

tumour source root: D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\data\zenodo\_3825933\extracted\val_dataset_1_tu
Dimensions: {"609x612": 113, "610x609": 64, "610x610": 61, "610x611": 123, "610x612": 52449}
Modes: {"RGB": 52810}

## VAL2

Dimensions (width x height): {"609x612": 131, "610x609": 69, "610x610": 59, "610x611": 141, "610x612": 33221, "611x609": 4, "611x610": 12, "611x611": 24, "611x612": 335, "612x612": 107}
Modes: {"RGB": 34103}

benign source root: D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\data\zenodo\_3825933\extracted\val_dataset_2_norm
Dimensions: {"609x612": 98, "610x609": 46, "610x610": 40, "610x611": 110, "610x612": 23739, "611x609": 4, "611x610": 11, "611x611": 22, "611x612": 304, "612x612": 97}
Modes: {"RGB": 24471}

tumour source root: D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\data\zenodo\_3825933\extracted\val_dataset_2_tu
Dimensions: {"609x612": 33, "610x609": 23, "610x610": 19, "610x611": 31, "610x612": 9482, "611x610": 1, "611x611": 2, "611x612": 31, "612x612": 10}
Modes: {"RGB": 9632}

## Evaluator compatibility

Current evaluate_paper.py present: False
Historical source reviewed: 1bf996d

Historical evaluator accepts archive roots via --benign-dir and --tumour-dir, recursively preserving norm/tu subdirectories.

Historical evaluator constructs its own manifest; it has no direct input-manifest option. A future CSV adapter would map class_name to source_group and ground_truth, width/height/mode to source_width/source_height/source_mode.

Stage 2B CSV fingerprints differ from evaluator internal CSV fingerprints because schemas differ. Do not pass these hashes as --expected-manifest-sha256.

Restore/review the evaluator in the intended branch before evaluation; no evaluator changes made.

Stage 2C data preparation can proceed. Evaluator execution remains pending restoration/review of the missing evaluator and its other prerequisites.
