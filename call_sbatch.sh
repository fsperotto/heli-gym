#!/usr/bin/env bash

# Usage: bash call_sbatch.sh [Python options]
# --n-envs also requests n-envs + 1 CPUs; all options reach the Python job.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "This script should not be sourced, since it simply launches jobs:" >&2
    echo "--> Use:  bash ./call_sbatch.sh" >&2
    return 1
fi

#stop after any error
set -euo pipefail

DATETIME=$(date '+%Y_%m_%d_%H_%M_%S')

PROJ_DIR="${WORKDIR:-$HOME}/SourceCode/GITHUB/heli-gym"
PROJ_URL="https://github.com/fsperotto/heli-gym.git"

# Only n-envs affects submission resources. Forward every Python argument intact.
# Keep the submission default (24) distinct from the standalone job default (7).
N_ENVS=24
ARGS=("$@")
while (($#)); do
    case "$1" in
        --n-envs)
            if (($# < 2)); then echo "Missing value for --n-envs" >&2; exit 2; fi
            N_ENVS=$2; shift 2 ;;
        --n-envs=*) N_ENVS=${1#*=}; shift ;;
        *) shift ;;
    esac
done
if [[ ! "$N_ENVS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--n-envs must be a positive integer" >&2
    exit 2
fi

if [[ ! -d "$PROJ_DIR" ]]; then
    export HTTP_PROXY=proxy.onera:80
    export HTTPS_PROXY=proxy.onera:80
    mkdir -p "$(dirname "$PROJ_DIR")"
    git clone "$PROJ_URL" "$PROJ_DIR"
fi

mkdir -p "${PROJ_DIR}/logs/$DATETIME"

sbatch \
    --job-name "heli-gym-learn-$DATETIME" \
    --chdir "$PROJ_DIR" \
    --cpus-per-task "$((N_ENVS + 1))" \
    --time 11:30:00 \
    --output "$PROJ_DIR/logs/$DATETIME/slurm.%j.out" \
    --error "$PROJ_DIR/logs/$DATETIME/slurm.%j.err" \
    "$PROJ_DIR/call_srun_learn_job.sh" \
        --n-envs "$N_ENVS" "${ARGS[@]}"
