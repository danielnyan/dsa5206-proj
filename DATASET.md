# Validation Dataset Setup

This repository does **not** store the original validation image archives because the dataset is approximately 15.3 GB compressed and is too large for normal GitHub storage.

Instead, team members should download the original validation datasets directly from the authors' Zenodo record and verify the files using the published MD5 checksums.

## Source

**Dataset:** Dataset Tolkach Y et al.
**Zenodo DOI:** 10.5281/zenodo.3825933
**Zenodo record:** https://zenodo.org/records/3825933
**Associated publication:** Tolkach et al., *High-accuracy prostate cancer pathology using deep learning*, Nature Machine Intelligence (2020)
**Publication DOI:** 10.1038/s42256-020-0200-7

The Zenodo record describes these files as the two validation datasets used in the publication.

## Validation Dataset Mapping

| Paper cohort | Class | Zenodo archive |
|---|---|---|
| VAL1 | Benign / normal | `val_dataset_1_norm.zip` |
| VAL1 | Tumour | `val_dataset_1_tu.zip` |
| VAL2 | Benign / normal | `val_dataset_2_norm.zip` |
| VAL2 | Tumour | `val_dataset_2_tu.zip` |

VAL1 and VAL2 must remain separate. Do **not** merge the two cohorts for the paper-comparison evaluation.

## Official Archive Checksums

| Archive | Approx. compressed size | Expected MD5 |
|---|---:|---|
| `val_dataset_1_norm.zip` | 7.4 GB | `1dda32f59996640d701b591f69353115` |
| `val_dataset_1_tu.zip` | 4.9 GB | `35494acbfea3adbc689cc15f73a9116d` |
| `val_dataset_2_norm.zip` | 2.0 GB | `029069ab405a76d42fcdefac32982989` |
| `val_dataset_2_tu.zip` | 937.1 MB | `d72b78d364fbfed95d8f0d71d8f2d01e` |

A dataset archive should not be used until its MD5 checksum matches the expected value above.

## Recommended Local Storage

The large dataset should be kept outside the Git repository.

Example Windows layout:

```text
D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\
├── GitHub\
│   └── dsa5206-proj\
└── data\
    └── zenodo_3825933\
        ├── val_dataset_1_norm.zip
        ├── val_dataset_1_tu.zip
        ├── val_dataset_2_norm.zip
        └── val_dataset_2_tu.zip
```

After extraction, keep VAL1 and VAL2 in separate directories.

A suitable structure is:

```text
data\
└── zenodo_3825933\
    ├── archives\
    │   ├── val_dataset_1_norm.zip
    │   ├── val_dataset_1_tu.zip
    │   ├── val_dataset_2_norm.zip
    │   └── val_dataset_2_tu.zip
    └── extracted\
        ├── VAL1\
        │   ├── benign\
        │   └── tumour\
        └── VAL2\
            ├── benign\
            └── tumour\
```

The exact extracted archive directory names may differ from this recommended layout. Do not rename or reorganize files until the original archive contents and image counts have been recorded.

## Download

From the repository root, the team download script can be run in PowerShell:

```powershell
.\tools\download_validation_data.ps1 `
    -DataDir "..\..\data\zenodo_3825933"
```

The script should:

1. Create the destination directory if necessary.
2. Download or resume each archive from Zenodo.
3. Skip an existing archive when its MD5 already matches.
4. Verify every completed download against the official MD5 checksum.
5. Stop with an error if verification fails.

## Manual Checksum Verification

To verify one archive manually in PowerShell:

```powershell
Get-FileHash `
    "D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\data\zenodo_3825933\val_dataset_2_tu.zip" `
    -Algorithm MD5
```

The hexadecimal result is case-insensitive.

To verify all four archives:

```powershell
Get-ChildItem `
    "D:\DSML 2025\Semester 1 (2026)\DSA5206\Project\data\zenodo_3825933" `
    -Filter "*.zip" |
    Get-FileHash -Algorithm MD5
```

## Reproducibility Rules for This Project

For the paper-reproduction work:

- Preserve the original downloaded archives.
- Record the MD5 checksum of each archive.
- Record extraction date and image counts.
- Keep VAL1 and VAL2 separate.
- Do not randomly split or combine VAL1 and VAL2 for the paper-comparison evaluation.
- Do not commit the image archives or extracted datasets to Git.
- Commit only scripts, manifests, checksums, documentation and evaluation code.
- Any experiment that trains on released validation data must be labelled as a **reimplementation**, not an exact reproduction of the authors' original trained model.

## Current Reproduction Limitation

The authors' original trained NASNetLarge prostate checkpoint and the complete original three-class training dataset are not available to this project. Therefore, the published native C1 accuracy cannot be reproduced directly from the released validation data alone.

The strict `modern_pca/evaluate_paper.py` evaluator is intended for paper-faithful native C1 evaluation when an appropriate authenticated three-output checkpoint is available.

For a constrained reimplementation, the planned experimental design is:

```text
VAL1
  -> development / training source
  -> internal validation as required

VAL2
  -> kept untouched
  -> final external test
```

Results from this fallback experiment must be compared with the publication as a **reimplementation result**, not described as exact reproduction.
