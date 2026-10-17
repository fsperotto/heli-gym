#!/usr/bin/env bash
#SBATCH --job-name=heligym_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --time=11:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/stck/fperotto/log/slurm.%j.out
#SBATCH --error=/stck/fperotto/log/slurm.%j.err

# _SBATCH --qos=co_long_gpu

set -euo pipefail

#srun python learn.py --no-wind --task=stabilize --n-steps=100000 --no-render --n-envs=24

#cd "$SLURM_SUBMIT_DIR"
cd "${WORKDIR:-/tmp_user/juno/${USER}}/SourceCode/heli-gym"

#activate the virtual environment
source ./init.sh
#source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

run_dir="$PWD/runs/juno-$SLURM_JOB_ID"
archive_dir="/stck/${USER}/SourceCode/heli-gym/juno-$SLURM_JOB_ID"
mkdir -p "$run_dir"

# Preserve available outputs after normal completion or a Python error.
archive_results() {
    status=$?
    trap - EXIT
    if ! mkdir -p "$archive_dir" ||
       ! cp -a "${run_dir}/." "${archive_dir}/"; then
        echo "Result archive failed; outputs remain in $run_dir" >&2
        status=1
    fi
    exit "$status"
}
#call the function on exit, even if the script is interrupted
trap archive_results EXIT

#save commit hash to know which version of the code was used for this run
git rev-parse HEAD > "${run_dir}/git-commit.txt"

#python -u is for disabling buffering output messages, so that they are printed in real time
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    .venv/bin/python -u learn.py \
    --task stabilize \
    --no-wind \
    --no-render \
    --device cuda \
    --torch-threads 1 \
    --n-envs 24 \
    --n-steps 1024 \
    --total-timesteps 1000000 \
    --output-dir "$run_dir"