# ESCNN D8: unattended CPU setup -> GPU training

## 1. Install scripts, not Python packages, from the login node

Unzip `escnn_overnight_pbs.zip` into the `extension-model` repository root.
Do NOT overwrite or recreate your Gleason environment or Stage-2G cache.
The scripts expect the six-file ESCNN D8 Stage-2G patch to be installed
and `requirements-group-equivariant.txt` to be present.

## 2. Edit `/scratch/YOUR_USERNAME/site.env` (outside Git)

The original `REPO_DIR` and `WORK_ROOT` remain as before. Add these lines,
replacing every placeholder with your actual path or a Python 3.11 module name:

```bash
ESCNN_STAGE2G_CACHE="$WORK_ROOT/path/to/verified/Stage2G/cache"
ESCNN_OUTPUT="$WORK_ROOT/runs/escnn-d8-stage2g-v1"
ESCNN_VENV_DIR="$WORK_ROOT/escnn-env"
# IMPORTANT: use your actual module name, discovered with: module avail Python
ESCNN_SITE_SETUP='source /app1/ebenv; module load Python/YOUR-PYTHON-3.11-MODULE'
# Or set ESCNN_PYTHON=/absolute/path/to/python3.11
ESCNN_GPU_JOBS=6
ESCNN_CHUNK_HOURS=42
# ESCNN_PBS_PROJECT=personal-YOUR_ACCOUNT  # only if required by your scheduler
# ESCNN_WHEELHOUSE=/shared/offline/python_wheels  # if compute nodes cannot access PyPI
```

`ESCNN_VENV_DIR` MUST be distinct from `VENV_DIR` (Gleason). The default
Gleason `SITE_SETUP` may load Python 3.13; `ESCNN_SITE_SETUP` overrides it
for both setup and training. The CPU job enforces Python 3.11.

## 3. Submit once from the login node

```bash
export PIPELINE_CONFIG=/scratch/YOUR_USERNAME/site.env
bash hpc/submit_escnn_overnight.sh
```

This submits one CPU job to validate or rebuild the venv, then six serialized
GPU jobs. Each GPU job is dependent on successful completion of the previous
job (`afterok`); it starts automatically, even after you disconnect.
The first GPU chunk performs a real D8 batch-4 forward/backward preflight and
reads a sample from the existing Stage-2G cache, then trains. The next
chunks resume `last.pt`; already-finished runs exit immediately.
The remaining GPU chunks are prequeued, not run concurrently.

A GPU chunk stops after ~42 hours, leaving six hours of walltime margin
under its 48-hour PBS request. Every few minutes the trainer commits
`last.pt`; `best.pt` holds the selected lowest VAL1 validation-loss model.
If the job reaches 20 epochs or four non-improving validation epochs, remaining
chunks exit promptly. If the setup or any chunk errors, later jobs should not
run, since dependencies use `afterok`.

## 4. Inspect when you return

```bash
qstat -u "$USER"
source /scratch/YOUR_USERNAME/site.env
cat "$WORK_ROOT/logs/escnn-overnight/cpu-setup.log"
tail -n 60 "$WORK_ROOT/logs/escnn-overnight/gpu-1.log"
ls -lh "$ESCNN_OUTPUT"/{last,best}.pt
```

PBS output redirection may be site-specific and may only appear when the job
finishes. Use `qstat -f JOBID` if logs are not yet visible.

## Cleanup/recovery policy

The CPU job first imports all required packages and runs `pip check`. If the
previous interrupted environment is healthy, it is reused. Otherwise it is
renamed to `escnn-env.incomplete.<PBS_JOBID>` and a clean environment is
created at the configured `ESCNN_VENV_DIR`. NOTHING is deleted automatically.
Do not delete the backup until the new environment succeeds and you no longer
need any files from the incomplete environment. The unrelated Gleason venv,
Stage-2G cache, and model checkpoints are never removed.

The login submit script refuses a second invocation when its marker exists.
To submit another chain later, only after confirming old jobs are finished,
change `ESCNN_GPU_JOBS` as desired and manually remove
`$WORK_ROOT/logs/escnn-overnight/.submitted` (an empty directory).

IMPORTANT: if compute nodes cannot access PyPI, CPU setup will fail; provide
an offline wheelhouse or seek site-approved software provisioning. If required
Python 3.11 / Fortran build tools aren't available, the CPU setup stops. If
PyTorch's CUDA wheel is incompatible with the GPU driver, GPU preflight fails
without starting training. These cases should not be auto-repaired by
changing the experimental model or batch size.

PBS `#PBS -P ...`, available queues, and maximum job count/walltime depend on
your site and must be adjusted if the scheduler rejects submission.
