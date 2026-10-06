# Prostate pathology — DSA5206

Two independent GP3/GP4/GP5 experiments adapt the legacy Tolkach NASNetLarge
implementation. No training data, teachers, or mined examples cross between tracks.

| Track | Sources | Preparation |
|---|---|---|
| `crowd-sicap` | CrowdGleason + SICAPv2 | Released patches, tumour-only labels |
| `gleason2019` | Gleason2019 only | Native 600px crops → 350px images |

## Run on the HPC

Follow [the HPC runbook](hpc/README.md) for one-time setup. After that, from the
repository directory:

```bash
export PIPELINE_CONFIG=/absolute/path/site.env

# Choose either track; downloads and preparation run in the CPU PBS job.
bash hpc/submit_pipeline.sh crowd-sicap smoke
bash hpc/submit_pipeline.sh gleason2019 smoke

# After checking that track's smoke run, train without repeating preparation.
bash hpc/submit_pipeline.sh crowd-sicap train
bash hpc/submit_pipeline.sh gleason2019 train
```

These are separate submissions; there is no need to run both at once.
The helper reuses completed preparation and keeps each track's outputs separate.
Full training is never automatically launched after a smoke run.

## Reference material

- [HPC commands and PBS jobs](hpc/README.md)
- [Implementation and inspection evidence](reports/gleason_implementation.md)
- [Earlier binary reimplementation](REIMPLEMENTATION.md)
- [Earlier validation datasets](DATASET.md)

Legacy folders and the earlier binary workflow remain intact, but are not the
GP3/GP4/GP5 HPC entry points.

## Assumptions and limitations

These are replacement-data experiments, not exact reproduction of the original
trained model. Architecture, preprocessing and the legacy training schedule are
preserved; stromal decontamination and unsupported expansion retraining are not
introduced. The automatic Crowd/SICAP route includes normalized SICAP images;
original SICAP paths can be supplied instead. Pure training requires its
`wsi_labels.xlsx`, which must be staged separately.

Gleason2019 uses available expert masks with pixel-level unique plurality,
including background/benign votes. Missing masks abstain; raw value 6 abstains
only at affected pixels; tied pixels remain unresolved. Uncertainty is recorded,
not a blanket patch/core exclusion. A single resolved tumour GP yields a patch
target; multiple GPs do not receive a combined or dominant-GP target.

Gleason2019's seed cohort uses single-pattern-core consensus as a documented
proxy, not verified pure-pattern patient cases. Splits are frozen before seed
selection/mining, approximately 80/20 by filename slide group with seed 42.
Patient independence is not established. Its full evaluation uses this internal
validation partition, not an invented labelled challenge test set.

Each track mines only its own multipattern training sources using its own
complete pure-cohort teacher and strict probability >0.95. Mining exports an
audited expansion pool; a subsequent retraining recipe is not implemented.

Differences in cohorts, annotation quality, seed-cohort construction and fields
of view mean this comparison cannot isolate resolution alone.
