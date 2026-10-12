# Shared settings for every NSCC ASPIRE 2A job and helper (sourced, not executed).
# Values marked VERIFY were not checked on the system yet; fix them after the first login
# (see README "First login: inspect the system").

# --- Account / scheduler -------------------------------------------------------------
: "${NSCC_PROJECT:=CHANGE_ME}"              # no project exists yet (2026-10-06); `qsub -P` needs one
: "${CPU_QUEUE:=normal}"                    # VERIFY: `qstat -Q`
: "${GPU_QUEUE:=normal}"                    # VERIFY: GPU jobs may need a dedicated queue (e.g. `ai`)

# --- Paths ----------------------------------------------------------------------------
# Lustre scratch for everything large; home is small. VERIFY the path and the purge policy.
: "${NSCC_ROOT:=/scratch/users/nus/${USER}/dsa5206}"
CODE="${CODE:-$NSCC_ROOT/code/dsa5206-proj}"         # git clone of the repository (LF endings)
ENV_DIR="${ENV_DIR:-$NSCC_ROOT/env/ens}"             # conda environment prefix
ARCHIVES="${ARCHIVES:-$NSCC_ROOT/zenodo_3825933}"    # the two VAL1 ZIPs only; VAL2 is never fetched
CACHE="${CACHE:-$NSCC_ROOT/ens_cache}"               # ens_cache_part_0 .. ens_cache_part_7
RUNS="${RUNS:-$NSCC_ROOT/runs}"
export KERAS_HOME="${KERAS_HOME:-$NSCC_ROOT/keras_home}"  # NASNetLarge ImageNet weights, fetched on the login node

# --- Software pins --------------------------------------------------------------------
# Empty = newest release pip resolves for this Python; setup_env.sh writes the resolved versions to
# $ENV_DIR/nscc_env_freeze.txt, every cache part records software(), and every checkpoint contract
# records tensorflow/keras. Pin a version here only to reproduce an earlier environment.
: "${PYTHON_VERSION:=3.13}"
: "${TENSORFLOW_VERSION:=}"
: "${KERAS_VERSION:=}"
: "${NUMPY_VERSION:=}"
: "${SCIPY_VERSION:=}"
: "${SKLEARN_VERSION:=}"

activate_env() {
    # VERIFY the module name (`module avail 2>&1 | grep -i -E 'conda|forge'`).
    module load miniforge3 2>/dev/null || module load anaconda3 2>/dev/null || true
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_DIR"
    # pip CUDA wheels: TensorFlow cannot dlopen them without this (verified by gpu_diag.pbs, 2026-10-08)
    export LD_LIBRARY_PATH="$(echo "$ENV_DIR"/lib/python3.*/site-packages/nvidia/*/lib | tr " " ":")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    cd "$CODE"
    export PYTHONPATH="$CODE" PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=1
}

session_note() {
    # One JSON line per job in $1/nscc_sessions.jsonl: what ran where (hardware is not in the contract).
    local out="$1" event="$2"; mkdir -p "$out"
    local gpu; gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)
    printf '{"event":"%s","utc":"%s","job":"%s","host":"%s","gpu":"%s","status":%s}\n' \
        "$event" "$(date -u +%FT%TZ)" "${PBS_JOBID:-interactive}" "$(hostname)" "$gpu" \
        "$(cat "$out/status.json" 2>/dev/null || echo null)" >> "$out/nscc_sessions.jsonl"
}
