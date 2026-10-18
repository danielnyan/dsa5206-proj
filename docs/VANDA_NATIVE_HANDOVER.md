# Vanda 14-epoch constrained NASNetLarge — team handover

**Scope:** This package contains only the **14-epoch VAL1-trained binary checkpoint handover**, plus the supporting inference procedure. It is **not** the original authors' three-class trained model and is not an exact reproduction. Keep unrelated experiments and downstream evaluation work outside this release.

## A. Producer account: audit before any push

On Vanda, from the project repository:

```bash
cd /scratch/e1536052/DSA5206/dsa5206-proj
pwd
git remote -v
git branch --show-current
git status --short
ls -lh /scratch/e1536052/DSA5206/vanda_runs/nasnet_production_14epoch/epoch14.keras
sha256sum /scratch/e1536052/DSA5206/vanda_runs/nasnet_production_14epoch/epoch14.keras
```

Expected SHA-256: `433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde`.

**Never stage the entire working tree with `git add .`, `git add -A`, or `git commit -a`.** Working trees may contain unrelated downstream experiments. Work in a dedicated branch based on the intended team base commit; review diffs and `git status` before commits. Verify whether project policy requires a pull request.

Recommended staging (from the repo root, after moving the prepared files into the proper locations):

```bash
git switch feat/native-epoch14-handover
# Add only reviewed handover files, plus any essential Vanda training source files confirmed by the team.
git add README.md docs/VANDA_NATIVE_HANDOVER.md modern_pca/vanda_native_infer.py modern_pca/vanda_distributed_training.py modern_pca/vanda_cache_consumer.py vanda_smoke/nasnet_14epoch.pbs vanda_smoke/nasnet_infer.pbs
git diff --cached --stat
git diff --cached --check
git diff --cached --name-only
# Examine the full staged diff before commit:
git diff --cached
git commit -m "docs(repro): share 14-epoch NASNetLarge Vanda checkpoint handover"
git push -u origin feat/native-epoch14-handover
```

This only stages the above three files. **If Vanda source files required by the team are not already in Git, identify and stage those explicitly after inspection.** A directory existing on Vanda does not imply it is committed. Do not add datasets, caches, `.venv`, TensorFlow logs, optimizer-state files or generated checkpoints to normal Git.

## B. GitHub checkpoint distribution — Release asset

The model is about 804 MiB, too large for a normal Git blob; publish it as a **GitHub Release asset** (GitHub Releases accept individual files under 2 GiB). Do **not** push the `.keras` file into standard Git history.

After the documentation/code PR is merged, determine the immutable source commit/tag. On a machine with GitHub CLI `gh` installed and authenticated with repository permissions:

```bash
# From the reviewed repository, at the intended release commit:
git switch main
git pull --ff-only
git status --short
# Proceed only if the checkout is clean and the correct code is present.
git tag -a vanda-native-epoch14-v1 -m "Verified 14-epoch binary NASNetLarge handover"
git push origin vanda-native-epoch14-v1

# In the repository directory, with the checkpoint available at CHECKPOINT_PATH:
CHECKPOINT_PATH=/PATH/TO/epoch14.keras
sha256sum "$CHECKPOINT_PATH"
gh release create vanda-native-epoch14-v1 "$CHECKPOINT_PATH" \
  --verify-tag --title "Vanda NASNetLarge 14 epochs — constrained binary checkpoint" \
  --notes "Research checkpoint; binary VAL1-trained NASNetLarge; SHA-256 433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde. Not the paper authors' checkpoint."
```

If `gh` is not installed on Vanda, perform release creation from an authenticated team workstation. You can copy the verified checkpoint from Vanda to an approved local or shared location using `scp` or institution-approved transfer tools. A GitHub personal access token should never be pasted into chats, committed to Git, or stored in shell history. A private repository's release assets still require authorized access.

Verify the release page and actual asset are accessible to teammates. Replace the README's `TO_BE_FILLED_AFTER_UPLOAD` with its URL in a follow-up review/commit.

## C. Teammate account: run the saved native-architecture model

1. Ensure your NUS account has Vanda compute privileges and accessible scratch storage. **Do not reuse the producer's user ID or assume their `/scratch/...` tree is readable.**
2. Clone the release-tagged repo through Git SSH or HTTPS.
3. Load the Python module and build a compatible private environment. The original execution used Python 3.11.5, TensorFlow 2.20, NumPy 1.26.4, Pillow 10.4 and legacy Macenko `staintools`/`spams` dependencies. The bundled `requirements-reimplementation-wsl.txt` is a Windows/WSL reference, **not a validated Vanda installer**. Installation of `spams` may require a separately configured environment. Ask the model producer for the working Vanda environment audit (`python -m pip freeze`, modules, CUDA visibility) before reinstalling; never copy another account's `.venv` blindly.
4. Download release asset `epoch14.keras` into your own scratch folder using an authenticated GitHub CLI or approved browser download/transfer method.
5. Hash-verify the checkpoint, ensure the legacy stain reference `4_WSI_pipeline/WSI_pipeline_v6/images/standard_he_stain_small.jpg` is present, and test the inference script on a **GPU compute node** (not the login node).

For a teammate with a correctly configured compute-node shell:

```bash
module load Python/3.11.5-GCCcore-13.2.0
source "$HOME/scratch/DSA5206/.venv/bin/activate"  # replace with YOUR validated venv path
cd "$HOME/scratch/DSA5206/dsa5206-proj"     # replace with YOUR cloned repo location
python -c 'import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices("GPU"))'
sha256sum "$HOME/scratch/DSA5206/checkpoints/epoch14.keras"
python -m modern_pca.vanda_native_infer \
  --model "$HOME/scratch/DSA5206/checkpoints/epoch14.keras" \
  --image /PATH/TO/PATHOLOGY_PATCH.jpg \
  --output /PATH/TO/OUTPUT/prediction.json
```

**GPU scheduling:** request a GPU node using the site's current PBS policy; do not execute the heavy 804-MiB TensorFlow model from the shared Vanda login node. Example PBS resources used by the original team: `#PBS -q auto`, `#PBS -l select=1:ncpus=12:mem=96gb:ngpus=1`. The actual allocation may be greater than requested and can vary by queue. Run the above commands *inside* the scheduled PBS job, or supply an institution-approved compute-node launcher.

**Interpreting output:** The standalone inference runner accepts one image patch, verifies image/model hashes and emits benign/tumour softmax probabilities at fixed threshold 0.5. It applies the original `preprocess_legacy` pipeline. It does not process whole-slide images, produce slide-level diagnoses, or replace the author's original checkpoint. The output is for research only.

## D. Full fresh training is separate from checkpoint inference

A full 14-epoch recreation requires the ~54 GB verified VAL1 cache, source JPEGs, immutable training manifests, external preprocessing dependencies, working two-A40 distributed training script and launch configuration, and several hours of GPU allocation. The GitHub Release publishes the **trained model for inference**, not the full data/cache/optimizer-state tree.

Before claiming that another team member can *retrain* the model end-to-end, verify that the actual production trainer (for example `modern_pca/vanda_distributed_training.py`) and its exact PBS launcher, cache-generation entry points, configuration and checksums are committed in the tagged repository. Do not invent a training CLI command; the team should document it only after checking the actual `--help` and successful Vanda launch script.

## E. Reproducibility and interpretation limits

- Training data: released **VAL1** binary patches, not the authors' unavailable annotated TCGA training corpus.
- VAL1 split is patch-disjoint but may not be patient- or slide-disjoint.
- Stain normalization reference and pinned software environment materially affect outputs.
- The original 14-epoch checkpoint must be distinguished from all subsequent adaptation variants.
- Keep the released artifact and source tag immutable. A changed file needs a new hash and release/version.

## F. Running native checkpoint inference on Vanda

Each teammate must use their own Vanda scratch directory,
compatible Python environment, and verified epoch14.keras checkpoint.

Submit inference from the Vanda login node:

    cd /scratch/$USER/DSA5206/dsa5206-proj
    qsub -v DSA5206_IMAGE=/scratch/$USER/DSA5206/input/test_patch.jpg vanda_smoke/nasnet_infer.pbs

Replace test_patch.jpg with an existing pathology patch.

The inference launcher defaults to:
- Model: /scratch/$USER/DSA5206/checkpoints/epoch14.keras
- Environment: /scratch/$USER/DSA5206/.venv
- Output: /scratch/$USER/DSA5206/inference/prediction.json

Use DSA5206_MODEL, DSA5206_VENV, DSA5206_REPO, or
DSA5206_OUTPUT to override the default paths.

Submit through PBS rather than running the large model
on the shared Vanda login node.
