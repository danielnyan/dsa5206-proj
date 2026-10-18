Vanda 2 x A40 14-epoch NASNetLarge trainer
==========================================
Copy vanda_distributed_training.py to:
  /scratch/e1536052/DSA5206/dsa5206-proj/modern_pca/vanda_distributed_training.py
Copy nasnet_14epoch.pbs to:
  /scratch/e1536052/DSA5206/dsa5206-proj/vanda_smoke/nasnet_14epoch.pbs

Before use: review the separate trainer against your full repository and
run a one-epoch pilot first. This code is syntactically checked but not
executed against Vanda or the repository in this environment.

On Vanda:
  cd /scratch/e1536052/DSA5206/dsa5206-proj
  python -m py_compile modern_pca/vanda_distributed_training.py
  bash -n vanda_smoke/nasnet_14epoch.pbs
  git diff -- modern_pca/reimplementation.py modern_pca/cached_training.py modern_pca/models.py
  qsub -v MAX_EPOCHS=1 vanda_smoke/nasnet_14epoch.pbs

After epoch 1 checkpoint is verified, resume through 14:
  qsub vanda_smoke/nasnet_14epoch.pbs

The fixed output directory is:
  /scratch/e1536052/DSA5206/vanda_runs/nasnet_production_14epoch
Each invocation auto-detects the latest valid contiguous epoch checkpoint.
Partial epochs are retrained; separate PBS submissions are still required.

If code changes after epoch 1, the strict code hash contract prevents an
unverified resume. Preserve the code from the pilot through production.
Checkpoint storage could require tens of GB for 14 epochs. Check quota.
