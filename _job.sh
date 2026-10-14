#!/usr/bin/env bash
#SBATCH --job-name=heligym_train
#SBATCH --ntasks=1
#SBATCH --time=0-12:00
#SBATCH --partition=gpu
#SBATCH --qos=co_long_gpu
#SBATCH --output=/stck/fperotto/log/slurm.%j.out
#SBATCH --error=/stck/fperotto/log/slurm.%j.err

#set -euo pipefail

source ./init.sh

srun python learn.py --no-wind --task=stabilize --n-steps=100000 --no-render --n-envs=24
