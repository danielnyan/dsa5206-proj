# PANDA preliminary Gleason training on Vanda — direct chunk ZIP workflow

This patch keeps the existing `gleason`-branch training path and PBS helper as
intact as possible. The main change is that `DATASET=panda` becomes a third
track. You upload the Kaggle transfer ZIPs directly to Vanda; the CPU prepare
job verifies and merges their contents into the existing manifest/cache contract.
There is no manual unzip step and no need to create one giant combined ZIP.

## 1. Apply the patch on the `gleason` branch

From the repository root:

```bash
# Copy the replacement/new HPC files from this bundle into the repo, preserving paths.
cp /path/to/bundle/hpc/submit_pipeline.sh hpc/submit_pipeline.sh
cp /path/to/bundle/hpc/site.env.example hpc/site.env.example
cp /path/to/bundle/hpc/prepare_panda_chunks.py hpc/prepare_panda_chunks.py
cp /path/to/bundle/hpc/pbs/prepare_gleason.pbs hpc/pbs/prepare_gleason.pbs
cp /path/to/bundle/hpc/pbs/train_gleason.pbs hpc/pbs/train_gleason.pbs
chmod +x hpc/submit_pipeline.sh hpc/prepare_panda_chunks.py hpc/pbs/*.pbs

git apply /path/to/bundle/modern_pca_panda.patch
```

Review `git diff` before submitting jobs.

## 2. Upload Kaggle chunk ZIPs directly

Example Vanda layout:

```text
/scratch/USER/raw/panda_chunks/
├── panda_150um_chunk_0000.zip
├── panda_150um_chunk_0000.zip.sha256.txt
├── panda_150um_chunk_0001.zip
├── panda_150um_chunk_0001.zip.sha256.txt
└── ...
```

The sidecars are required. `prepare_panda_chunks.py` verifies the whole ZIP,
then checks the manifest hashes inside the ZIP and the SHA-256 of every PNG as
it streams the file into the prepared cache.

## 3. Configure a preliminary one/few-chunk cohort

Copy `hpc/site.env.example` outside the repository as before and edit at least:

```bash
REPO_DIR=/scratch/USER/dsa5206-proj
WORK_ROOT=/scratch/USER/dsa5206-work
PANDA_CHUNKS_DIR=/scratch/USER/raw/panda_chunks
PANDA_ALLOW_PARTIAL=1

# Freeze this path for the exact set of chunks currently uploaded.
PREPARED_DIR=/scratch/USER/dsa5206-work/prepared/panda-prelim-chunk0000
RUN_ROOT=/scratch/USER/dsa5206-work/runs/panda-prelim-chunk0000

SMOKE_BATCH_SIZE=1
BATCH_SIZE=4
```

Use a **new `PREPARED_DIR` and `RUN_ROOT` if you later add more chunks**. The
submit helper deliberately refuses to silently extend an already prepared
cohort because that would destroy reproducibility of the first preliminary run.

## 4. One-time environment bootstrap

```bash
export PIPELINE_CONFIG=/scratch/USER/site.env
bash hpc/bootstrap_env.sh
```

## 5. Prepare + smoke using the existing submit helper

```bash
bash hpc/submit_pipeline.sh panda smoke
```

This performs:

```text
uploaded chunk ZIPs
  -> CPU PBS preparation
  -> ZIP + manifest + PNG checksum verification
  -> PREPARED_DIR/manifests/panda_train.csv
  -> PREPARED_DIR/manifests/panda_validation.csv
  -> PREPARED_DIR/cache/*.png
  -> download/stage NASNetLarge ImageNet weights
  -> dependent GPU smoke job
```

Inspect the smoke output under `RUN_ROOT/<PBS_JOBID>-smoke`. A successful smoke
has `run.json` with `complete: true`, two completed smoke stages, and
`final.keras`.

## 6. Preliminary 14-epoch run

After smoke succeeds:

```bash
bash hpc/submit_pipeline.sh panda train
```

For the partial chunk cohort, `PANDA_ALLOW_PARTIAL=1` causes the PBS wrapper to
pass the explicit `--allow-sample-only` opt-in. `run.json` records the model as
preliminary/sample-only and records any non-100 physical batch as a deviation.

The underlying architecture/schedule stays the existing branch's three-class
Gleason trainer:

```text
350x350 RGB
 -> ImageNet NASNetLarge
 -> Flatten
 -> Dense(256, ReLU)
 -> Dense(3, softmax): GP3 / GP4 / GP5
```

Schedule: 7 epochs from `normal_conv_1_17` at 1e-5, then one epoch each from
boundaries 14, 11, 9, 7, 5, 3, 1 at 1e-6 (14 epochs total).

`BATCH_SIZE=4` is a practical starting point for the very large Flatten-head
NASNet. It is a documented methodological deviation from the branch's legacy
physical batch 100. If it OOMs, try 2 or 1.

## 7. When more chunks arrive

Do not modify the old prepared cohort. Upload the additional ZIP + sidecar files,
then choose a fresh cohort path, for example:

```bash
PREPARED_DIR=/scratch/USER/dsa5206-work/prepared/panda-prelim-chunks0000-0002
RUN_ROOT=/scratch/USER/dsa5206-work/runs/panda-prelim-chunks0000-0002
PANDA_ALLOW_PARTIAL=1
```

Run smoke and train again.

For the eventual full cohort (all contiguous chunks through the shard whose
`chunk.json` has `is_last_chunk=true`), use another fresh path and disable the
partial escape hatch:

```bash
PANDA_ALLOW_PARTIAL=0
PREPARED_DIR=/scratch/USER/dsa5206-work/prepared/panda-full-v1
RUN_ROOT=/scratch/USER/dsa5206-work/runs/panda-full-v1
bash hpc/submit_pipeline.sh panda smoke
bash hpc/submit_pipeline.sh panda train
```

With `PANDA_ALLOW_PARTIAL=0`, preparation refuses gaps, missing chunk 0, or a
cohort that does not reach the final shard.

## 8. Important interpretation

A model trained from only the first chunk(s) is a **preliminary/bounded PANDA
transfer experiment**, not the final full-cohort Gleason model and not an exact
reproduction of the original Tolkach training cohort. Keep its model directory
and prepared directory immutable so later comparisons are traceable.
