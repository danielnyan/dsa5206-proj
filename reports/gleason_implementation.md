# Tumour-only Gleason implementation handoff

Implemented against upstream base `364082e`; original legacy scripts and their
examples are retained. Start with [the HPC guide](../hpc/README.md).

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

## Verification actually performed on this VM

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
  and no unreleased confidence-expansion or replay curriculum.
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
