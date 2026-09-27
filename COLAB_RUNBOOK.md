# Tolkach prostate-pathology Colab runbook

This branch preserves the 2020 author code and adds an executable TensorFlow 2 /
Keras 3 pipeline in `modern_pca`. The modern path uses public APIs supported by
current Colab runtimes; it does not silently invoke TensorFlow 1 compatibility.

## Reproduction boundary

An exact full-paper reproduction is not possible from the public artifacts.
The GitHub repository has source and small examples but no trained weights. The
paper's TCGA WSIs are public, but the author-drawn annotation masks, the roughly
1.5 million extracted training patches, Gleason validation images/clinical
labels, and final models are not released. Zenodo record 3825933 releases only
the two tumour-versus-benign validation cohorts (four ZIPs, 15.3 GB).

Consequently, keep these claims separate:

1. **Code/runtime reproduction:** build the exact NASNetLarge + Flatten +
   Dense(256) architecture, execute its original 14-epoch progressive-unfreezing
   schedule, and run the C1/C8 algorithms.
2. **Released-data experiment:** retrain/evaluate a binary classifier on a new
   split of the released validation patches. This is useful for feasibility and
   architecture comparisons, but it is not the paper's independent validation.
3. **Exact result reproduction:** blocked until the authors supply final weights
   or the annotated training patches plus missing Gleason/clinical validation
   artifacts.

## Colab execution

1. Select a GPU runtime and upload `deep_learning_pca_modern.zip` plus
   `Prostate_Pathology_Colab_Runbook.ipynb`.
2. Open the notebook and run all cells in `examples` mode. This performs real
   CPU and GPU forward/backward passes, one stage of training, held-out
   validation, model serialization, reload, and C1/C8 evaluation using author
   example histology images.
3. Set `RUN_MODE = "released_full"` to download all four Zenodo archives with
   resumable `curl`, verify their MD5 checksums, extract them, and execute the
   same pipeline on the complete released dataset. This mode applies Macenko
   brightness/stain normalization using the author's bundled reference image.
4. Set `RUN_PAPER_SCHEDULE = True` only after measuring the one-epoch rate. It
   runs all original stages: 7 epochs releasing NASNet cells 17--18, followed by
   seven one-epoch stages progressively releasing cells 14 through 1.
5. Add `resnet50` or `efficientnetv2b0` to `ALTERNATIVE_ARCHITECTURES` for fair
   runs on the identical split. Alternatives use the decision schedule because
   NASNet layer boundaries do not apply to them.

## Expected resource pressure

The exact paper head uses `Flatten` rather than global average pooling and has a
very large dense layer. Start with batch size 1--4 on a T4. An L4 or A100 is
preferable for the complete 350 px dataset. `head="gap"` is provided as an
explicit alternative architecture, never as an exact reproduction.

The four public archives require 15.3 GB compressed plus extraction space.
Colab ephemeral storage should have at least 40 GB free before download.

## Direct command-line use

```bash
python -m modern_pca.train \
  --data-root /content/data/classes \
  --output /content/runs/nasnet_paper \
  --architecture nasnetlarge --head original --weights imagenet \
  --schedule paper --batch-size 4

python -m modern_pca.evaluate \
  --model /content/runs/nasnet_paper/final.keras \
  --data-root /content/data/classes \
  --output /content/runs/nasnet_paper/evaluation
```

For CPU benchmarking, start a fresh process so TensorFlow cannot place variables
on the GPU:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m modern_pca.benchmark \
  --data-root /content/data/classes --architecture nasnetlarge --head original
```
