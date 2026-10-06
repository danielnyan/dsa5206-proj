# Tumour-only Gleason implementation handoff

Original implementation base `364082e`; this revision is on `gleason`, based on
`6744af2`. Original legacy scripts/examples are retained. Start with
[the HPC guide](../hpc/README.md).

## Pure-source / confidence-mining revision

SICAP sample inspection confirms 155 slide-level primary/secondary annotation
rows in `wsi_labels.xlsx`; these are not paired patch targets. Same-pattern slide
grades plus matching published patch classes define the documented pure-source
initial cohort. This is a proxy for original region purity, not an assertion of
equivalent expert annotation. Multipattern, conflicting and unknown source roles
are recorded separately. No source annotation is overwritten. Crowd MV remains
available in the explicitly named `published` baseline; absent pure-region
metadata it is not admitted to the pure initial-model cohort.

Optional `MODE=mine` in the existing evaluation PBS job runs the frozen, complete,
non-smoke pure initial model on multipattern training candidates only. Selection
is strictly `max(p_gp3,p_gp4,p_gp5) > 0.95`; exact-threshold predictions are
rejected, and retained targets remain GP3/GP4/GP5. Original labels, all candidate
probabilities, source identities, teacher hash, threshold and artifact hashes
are preserved. Known held-out sample/content/slide/patient overlap aborts pure
training and mining. Older/published/smoke teachers and changed manifests fail.

The paper supports confidence mining, contrary to the first-cut blanket exclusion.
The available methods do not fully specify subsequent training scheduling, so
no automatic retraining, mixing ratio, replay or repeated rounds are invented.
The deliverable is an audited expansion pool, not a trained expanded model.

The old checked-in `RELEASE.json` was removed because its hashes describe the
previous archive. It is recoverable from Git. The packaging tool now generates
one fresh release manifest per ZIP and excludes that manifest and deleted source files
from that manifest. No old legacy procedures outside the Gleason branch change
set were deleted. The previous Blob ZIP is not updated by this local revision.

### Revision checks

`tests/test_gleason.py` and `tests/test_modern_static.py`: **44 passed, 1 skipped**
in the small CPU test environment. The skip is the optional native StainTools
pixel-parity test (not installed in this revision environment; see historical
results below). Coverage includes all three threshold boundaries, invalid
probabilities, source roles, held-out identity rejection, teacher/manifest
eligibility, retained/rejected provenance, empty-selection exports, mocked
end-to-end mining, and regenerated archive manifests. Python parsing, CLI help,
shell/PBS syntax and whitespace checks passed. Mocked inference is not GPU
validation; no actual model training, mining inference or PBS submission ran.

Joining the supplied SICAP slide workbook to the normalized split annotations
without loading images produced **1,026 pure-source training rows** (268 GP3,
450 GP4, 308 GP5), **3,299 multipattern training candidates**, and 59 conflicting
training rows. The conflicting rows are preserved, not silently relabelled or
deleted. Known SICAP slide/patient identities did not cross held-out partitions
for the eligible seed/candidate sets. Image-content checks still require the full
downloaded dataset. These are metadata eligibility counts, not accepted mining
counts or a guarantee of histological purity.

## Delivered

- Resumable, checksum-checked public data and NASNet ImageNet weight downloader.
- CPU preparation: published CrowdGleason majority vote (`MV`), curated test
  ground truth, published SICAP patch classes, NC exclusion, explicit
  `GP3/GP4/GP5` mapping, source/split/content audit, lossless normalized caches,
  and checksum-bound manifests. Marker votes and SICAP G4C remain metadata.
- Existing StainTools 2.1.2 brightness/Macenko adapter and fixed reference reused.
- NASNetLarge/Flatten/Dense256/Dense3 training, legacy 14-epoch release schedule,
  Adam resets, categorical loss, flips, and propagated BatchNorm training mode.
  Completed-stage restart, bounded GPU smoke checks, and model provenance added.
- Held-out patch classification reported separately for the two datasets.
- Region aggregation and original scoring function, bounded stability sampling,
  and WSI C1/C8/environment/Type-2 wrappers. A compatible, separately supplied
  tumour detector is still required for WSI use.
- Six PBS jobs, environment configuration, dependency-aware submission, and
  source ZIP packaging with per-file SHA-256 verification.

The only modification to an existing Python module is an optional backbone
training-mode argument in `modern_pca/models.py`; its existing callers retain
their previous default. New grading entrypoints are separate from the preexisting
binary reimplementation. Root README and ignore rules are also updated.

## First-cut verification (historical, before confidence mining)

Small CPU-only VM: approximately 841 MiB RAM and one CPU. No full TensorFlow
installation, NASNet model construction, full data download, or training was
attempted here.

```text
tests/test_evaluate_paper.py + tests/test_gleason.py: 123 passed, 23 skipped
tests/test_modern_static.py:                         2 passed
New/changed Python files:                          9 parsed successfully
Shell/PBS scripts:                                 9 passed bash -n
New command-line entrypoints:                      5 passed --help
git diff --check:                                  passed
```

The 23 skips require TensorFlow. Passing tests cover published-label handling,
NC removal in both datasets, class ordering, split/content leakage checks, safe
ZIP extraction, manifest tampering, legacy score thresholds/ties, soft-probability
aggregation, eight unique D4 views, bounded stability sampling, and actual native
StainTools normalization. Cached pixel values matched the established adapter
exactly in the normalization test.

The complete preparation command was also run on the supplied SICAP sample and
Crowd annotations, with `--sample-only --workers 1`. It normalized all 10 supplied
SICAP images and wrote six manifests plus cache/provenance: 3 GP4 and 7 GP5
training patches. There were 31,148 missing image joins from the full annotation
tables, expected for these samples. No Crowd image sample was available locally.
This is a preparation check, not training data: it lacks GP3 and validation/test
images. Production training/final evaluation reject sample-only preparations.
Sample data and temporary outputs are not included in the source ZIP.

Local native checks used Python 3.11.15, StainTools 2.1.2, spams-bin 2.6.13,
NumPy 1.26.4, Pillow 10.4.0, OpenCV 4.11.0.86, SciPy 1.17.1, and OpenPyXL 3.1.5.
The cluster environment starts from the existing pinned repository requirements.

## Explicit scientific limitations

- No stromal decontamination or substitute, no invented purity filters/crops,
  and no invented confidence-expansion training or replay curriculum. The
  documented confidence-mining selection is now implemented separately.
- Original training patches/checkpoints are unavailable. Replacement public
  patches have different labels and physical fields of view; resizing does not
  reconstruct the original sampling area.
- The automated source route supplies pre-normalized SICAPv2. Acceptance is
  explicit and recorded; established preprocessing is subsequently applied.
  Original SICAP data can instead be supplied through the four source paths.
- Supplied splits are retained. Slide-prefix overlaps are reported; patient
  independence cannot be established solely from those prefixes. Identical
  image content across partitions aborts preparation.
- Smaller physical batches change BatchNorm statistics. Batch 100 remains the
  production default, not a claim that every GPU can accommodate it.
- Region/slide predictions are not a validated clinical system. Reference
  region/slide labels and suitable physical resolution are needed for those
  evaluation comparisons; patch labels alone do not supply them.

## Required cluster execution (not performed)

1. Set site paths, modules, PBS queue/account/GPU syntax and scratch allocation.
2. Install the environment and download/checksum archives and ImageNet weights
   on permitted transfer resources; extract/prepare in the CPU PBS job.
3. Run the GPU smoke job. Check finite computation, trainability, actual weight
   updates, save/reload, RAM/VRAM, throughput and storage use.
4. Select suitable resources/physical batch, submit the full 14-epoch job, then
   run held-out evaluation on its frozen checkpoint.
5. Exercise region/stability/WSI jobs only with appropriate real inputs and a
   compatible tumour gate; record field of view and domain limitations.

PBS syntax checks are not scheduler execution. GPU training, GPU serialization,
and model-backed region/WSI inference remain unverified until those jobs run.
# Gleason2019 feasibility and HPC preparation (2026-10-06)

Inspection downloaded all 244 official training JPEGs (997,451,947 bytes).
The annotation mirror is 49,635,529 bytes with SHA256
`feba9d1620d4dec1926ae142ced7afaa91d028630a6fb160b0c110e36b95625a`.
It contains masks/documentation, **not photographs**. Decoding all six nested
expert archives found 1,182 masks spanning 247 core identities, 20 all-zero
masks, and raw categorical values 0, 1, 3, 4, 5, 6. Expert coverage is uneven;
absent expert masks are not absent image labels. Values at most 6 appear nearly
black in ordinary grayscale viewers. These are valid numeric segmentation maps,
although zero masks and unsupported 6 must not silently become tumour targets.
Every one of the 244 training images has at least three masks: 5 have three,
79 have four, 120 have five, and 40 have six. Three mask-only core identities
are recorded as orphans rather than matched to invented images.

Sources: [official download page](https://gleason2019.grand-challenge.org/Register/),
[annotation mirror](https://data.mendeley.com/datasets/s384w7kv78/1),
[author's dataset handling](https://github.com/adfoucart/deephisto/blob/master/gleason/dataset.py).
The author's separate implementation remaps values greater than 5 to benign;
we do not silently import that correction or its contour exclusions.

The CPU PBS route now automatically downloads, joins, cuts native 600px tiles
and prepares 350px legacy-normalized tumour patches. It outputs an audited
unsplit corpus, not an asserted reproduction training cohort. Pixel plurality
including background is an explicit preparation policy, not provided
challenge ground truth; ties/value-6 tiles are unresolved and excluded. Pure
patch appearance does not establish pure-case eligibility. No model architecture,
training schedule, mixed-GP pseudo-class, or stromal method is added.

Verification: 47 tests passed, one native-StainTools-dependent test skipped;
Python CLI/compilation and PBS shell syntax checks passed. Synthetic end-to-end
tests cover voting, exclusions, coordinates, edge accounting, and 350px cache
output (identity-normalizer fixture, not a claim of real stain normalization).
The implemented downloader successfully queried the live official service and
checksum-verified all 244 existing temporary JPEGs through its resume path.
Full production normalization/PBS execution was not run on this small VM.
