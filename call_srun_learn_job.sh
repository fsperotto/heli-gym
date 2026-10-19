#!/usr/bin/env bash

## default SBATCH options for the job
#SBATCH --job-name=heligym_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --time=11:30:00
#SBATCH --partition=intel
#SBATCH --output=/stck/fperotto/log/slurm.%j.out
#SBATCH --error=/stck/fperotto/log/slurm.%j.err

# _SBATCH --partition=gpu
# _SBATCH --gres=gpu:1
# _SBATCH --qos=co_long_gpu

set -euo pipefail

echo "HELI-GYM training job started at $(date) on $(hostname)"

#srun python main.py train --no-wind --task=stabilize --n-steps=100000 --n-envs=24

MACHINE="juno"

JOB_ID=${MACHINE}-${SLURM_JOB_ID:-$$}-$(date +%Y-%m-%d_%H-%M-%S)

#PROJECT_DIR="$SLURM_SUBMIT_DIR"
PROJECT_DIR="/stck/${USER}/SourceCode/GITHUB/heli-gym"
WORKING_DIR="${WORKDIR:-/tmp_user/${MACHINE}/${USER}}/SourceCode/GITHUB/heli-gym"
RUNNING_DIR="$WORKING_DIR/runs/$JOB_ID"
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
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    .venv/bin/python -u main.py train eval plot \
    --task stabilize \
    --no-wind \
    --device cuda \
    --torch-threads 1 \
    --n-envs 24 \
    --n-steps 1024 \
    --total-timesteps 1000000 \
    --target-dir "$RUNNING_DIR"