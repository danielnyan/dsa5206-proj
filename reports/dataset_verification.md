# Stage 2A dataset verification

Source: Zenodo record 3825933. Checksums and mapping supplied by the user.

UTC: 2026-10-01T04:49:35.527600+00:00

Requested source: D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\data\zenodo\_3825933
Actual archive directory: D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\data\zenodo_3825933
Extraction root: D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\data\zenodo\_3825933\extracted

Archives found in data/zenodo_3825933, matching DATASET.md; extraction destination kept exactly as requested.

All four MD5 checks passed: **True**

| Archive | Cohort/class | Exact ZIP bytes | MD5 result | Actual MD5 |
|---|---|---:|---|---|
| val_dataset_1_norm.zip | VAL1/benign | 7422740863 | PASS | 1dda32f59996640d701b591f69353115 |
| val_dataset_1_tu.zip | VAL1/tumour | 4934331128 | PASS | 35494acbfea3adbc689cc15f73a9116d |
| val_dataset_2_norm.zip | VAL2/benign | 1973499064 | PASS | 029069ab405a76d42fcdefac32982989 |
| val_dataset_2_tu.zip | VAL2/tumour | 937060246 | PASS | d72b78d364fbfed95d8f0d71d8f2d01e |

## Extracted inventory

| Archive | Total files | Images (recursive) | Non-images |
|---|---:|---:|---:|
| val_dataset_1_norm.zip | 93009 | 93009 | 0 |
| val_dataset_1_tu.zip | 52810 | 52810 | 0 |
| val_dataset_2_norm.zip | 24471 | 24471 | 0 |
| val_dataset_2_tu.zip | 9632 | 9632 | 0 |

## Verification summary

- VAL1: 145,819 images (93,009 benign; 52,810 tumour).
- VAL2: 34,103 images (24,471 benign; 9,632 tumour).
- Total: 179,922 readable RGB JPEGs; zero non-image files.
- No zero-byte files, unreadable images, case-only path collisions, partial download files, duplicate archives, or SHA-256 content duplicates were found.
- Every extracted relative file path matches its source ZIP member; ZIP CRC and extracted counts passed.
- The only image anomaly is native size variation: 1,837 images differ from dominant 610x612 (width x height). All are readable RGB images, with width/height between 609 and 612 pixels. No image was resized or removed.

| Dataset | Dominant dimensions | Images with other dimensions | Mode deviations |
|---|---|---:|---:|
| VAL1 benign | 610x612 | 594 | 0 |
| VAL1 tumour | 610x612 | 361 | 0 |
| VAL2 benign | 610x612 | 732 | 0 |
| VAL2 tumour | 610x612 | 150 | 0 |

The JSON report identifies every dimension-variant image by archive-relative path and records its dimensions and mode.

## Anomalies

- unexpected_initial_entries: 0
- non_image_files: 0
- zero_byte_files: 0
- unreadable_images: 0
- case_only_collision_groups: 0
- within_archive_duplicate_groups: 0
- dimension_mode_outliers: 1837
- image_warnings: 0
- multi_frame_images: 0
- VAL1_benign_vs_tumour_duplicate_groups: 0
- VAL2_benign_vs_tumour_duplicate_groups: 0
- VAL1_vs_VAL2_duplicate_groups: 0

Duplicate contents are SHA-256 exact-byte matches, not perceptual matches. The JSON lists every duplicate group and affected path. Nothing was deleted.

### val_dataset_1_norm.zip

Extraction: D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\data\zenodo\_3825933\extracted\val_dataset_1_norm

Image extensions: {".jpg": 93009}
Dimensions (width x height): {"609x612": 209, "610x609": 70, "610x610": 86, "610x611": 229, "610x612": 92415}
Modes: {"RGB": 93009}
Within-archive duplicate groups: 0

First five relative image paths (case-sensitive Unicode lexical order):

- norm/norm.0.jpg
- norm/norm.1.jpg
- norm/norm.10.jpg
- norm/norm.100.jpg
- norm/norm.1000.jpg

Last five relative image paths:

- norm/norm.9995.jpg
- norm/norm.9996.jpg
- norm/norm.9997.jpg
- norm/norm.9998.jpg
- norm/norm.9999.jpg

### val_dataset_1_tu.zip

Extraction: D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\data\zenodo\_3825933\extracted\val_dataset_1_tu

Image extensions: {".jpg": 52810}
Dimensions (width x height): {"609x612": 113, "610x609": 64, "610x610": 61, "610x611": 123, "610x612": 52449}
Modes: {"RGB": 52810}
Within-archive duplicate groups: 0

First five relative image paths (case-sensitive Unicode lexical order):

- tu/tum.0.jpg
- tu/tum.1.jpg
- tu/tum.10.jpg
- tu/tum.100.jpg
- tu/tum.1000.jpg

Last five relative image paths:

- tu/tum.9995.jpg
- tu/tum.9996.jpg
- tu/tum.9997.jpg
- tu/tum.9998.jpg
- tu/tum.9999.jpg

### val_dataset_2_norm.zip

Extraction: D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\data\zenodo\_3825933\extracted\val_dataset_2_norm

Image extensions: {".jpg": 24471}
Dimensions (width x height): {"609x612": 98, "610x609": 46, "610x610": 40, "610x611": 110, "610x612": 23739, "611x609": 4, "611x610": 11, "611x611": 22, "611x612": 304, "612x612": 97}
Modes: {"RGB": 24471}
Within-archive duplicate groups: 0

First five relative image paths (case-sensitive Unicode lexical order):

- norm/norm.0.jpg
- norm/norm.1.jpg
- norm/norm.10.jpg
- norm/norm.100.jpg
- norm/norm.1000.jpg

Last five relative image paths:

- norm/norm.9995.jpg
- norm/norm.9996.jpg
- norm/norm.9997.jpg
- norm/norm.9998.jpg
- norm/norm.9999.jpg

### val_dataset_2_tu.zip

Extraction: D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\data\zenodo\_3825933\extracted\val_dataset_2_tu

Image extensions: {".jpg": 9632}
Dimensions (width x height): {"609x612": 33, "610x609": 23, "610x610": 19, "610x611": 31, "610x612": 9482, "611x610": 1, "611x611": 2, "611x612": 31, "612x612": 10}
Modes: {"RGB": 9632}
Within-archive duplicate groups: 0

First five relative image paths (case-sensitive Unicode lexical order):

- tu/tu.0.jpg
- tu/tu.1.jpg
- tu/tu.10.jpg
- tu/tu.100.jpg
- tu/tu.1000.jpg

Last five relative image paths:

- tu/tu.995.jpg
- tu/tu.996.jpg
- tu/tu.997.jpg
- tu/tu.998.jpg
- tu/tu.999.jpg

## Separation and Stage 2B

VAL1 and VAL2 were not merged, randomly split, renamed or reorganized. Four independent extraction roots preserve archive-internal paths. Physical separation does not prove absence of shared contents; see SHA-256 findings.

**Stage 2B: READY_WITH_REVIEW**
Stage 2B manifest generation can proceed with VAL1 and VAL2 kept separate. Record and retain all 1,837 native-dimension variants; do not resize, discard or relabel original images. No exact-byte duplicates or unreadable files were found.

This does not authenticate patient-level independence, clinical labels, or checkpoint/preprocessing parity.
