#!/usr/bin/env bash

# Usage: bash call_srun_learn_job.sh [Python options]
# Runs train/eval/plot; explicit Python options override launcher defaults.

## default SBATCH options for the job
#SBATCH --job-name=heligym_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=11:30:00
#SBATCH --partition=gpu
#SBATCH --output=/stck/fperotto/slurm_logs/slurm.%j.out
#SBATCH --error=/stck/fperotto/slurm_logs/slurm.%j.err

# _SBATCH --partition=gpu
# _SBATCH --gres=gpu:1
# _SBATCH --qos=co_long_gpu

set -euo pipefail

HOSTNAME=${HOSTNAME:-$(hostname)}

echo "HELI-GYM training job started at $(date) on ${HOSTNAME}"

# Python supplies its own defaults for other options. Launcher defaults precede
# user arguments so explicit values win. Wind flags are mutually exclusive.
DEFAULT_ARGS=(--task hover --n-envs 7 --total-timesteps 1000000
              --device cuda --torch-threads 1 --n-steps 1024)
HAS_WIND_FLAG=false
for arg in "$@"; do
    case "$arg" in --wind|--no-wind) HAS_WIND_FLAG=true ;; esac
done
if [[ "$HAS_WIND_FLAG" == false ]]; then DEFAULT_ARGS+=(--no-wind); fi

JOB_ID=${HOSTNAME}-${SLURM_JOB_ID:-$$}-$(date +%Y-%m-%d_%H-%M-%S)
echo "Job ID: $JOB_ID"

#PROJECT_DIR="$SLURM_SUBMIT_DIR"
#PROJECT_DIR="/stck/${USER}/SourceCode/GITHUB/heli-gym"
PROJECT_DIR="${HOME}/SourceCode/GITHUB/heli-gym"
#WORKING_DIR="${/tmp_user/${HOSTNAME}/${USER}}/SourceCode/GITHUB/heli-gym"
WORKING_DIR="${WORKDIR:-$HOME}/SourceCode/GITHUB/heli-gym"
RUNNING_DIR="${WORKING_DIR}/runs/$JOB_ID"
ARCHIVE_DIR="${PROJECT_DIR}/runs/$JOB_ID"

echo ;
echo "Project directory: $PROJECT_DIR"
echo "Working directory: $WORKING_DIR"
echo "Running directory: $RUNNING_DIR"
echo "Archive directory: $ARCHIVE_DIR"

cd "${WORKING_DIR}"

#activate the virtual environment
source ./init.sh
#source .venv/bin/activate

echo ;
echo "Using $(python --version) from $(which python)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

#when using matplotlib, cluster jobs only save figures, without showing them in a window.
export MPLBACKEND=Agg

mkdir -p "$RUNNING_DIR"

# Preserve available outputs after normal completion or a Python error.
archive_results() {
    status=$?
    trap - EXIT
    if ! mkdir -p "$ARCHIVE_DIR" ||
       ! cp -a "${RUNNING_DIR}/." "${ARCHIVE_DIR}/"; then
        echo "Result archive failed; outputs remain in $RUNNING_DIR" >&2
        status=1
    fi
    exit "$status"
}
#call the function on exit, even if the script is interrupted
trap archive_results EXIT

#save commit hash to know which version of the code was used for this run
git rev-parse HEAD > "${RUNNING_DIR}/git-commit.txt"

#python -u is for disabling buffering output messages, so that they are printed in real time
srun \
    --ntasks 1 \
    --cpus-per-task "${SLURM_CPUS_PER_TASK:-8}" \
    .venv/bin/python -u main.py train eval plot \
        "${DEFAULT_ARGS[@]}" \
        --target-dir "$RUNNING_DIR" \
        "$@"
