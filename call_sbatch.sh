#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "This script should not be sourced, since it simply launches jobs:" >&2
    echo "--> Use:  bash ./call_sbatch.sh" >&2
    return 1
fi

#stop after any error
set -euo pipefail

DATETIME=$(date '+%Y_%m_%d_%H_%M_%S')

PROJ_DIR="${WORKDIR}/SourceCode/GITHUB/heli-gym}"

if [[ ! -d "$PROJ_DIR" ]]; then
    export HTTP_PROXY=proxy.onera:80
    export HTTPS_PROXY=proxy.onera:80
    mkdir -p "$(dirname "$PROJ_DIR")"
    git clone https://github.com/fsperotto/heli-gym.git "$PROJ_DIR"
fi

mkdir -p "${PROJ_DIR}/logs/$DATETIME"

sbatch \
    --chdir="$PROJ_DIR" \
    --output="$PROJ_DIR/logs/$DATETIME/slurm.%j.out" \
    --error="$PROJ_DIR/logs/$DATETIME/slurm.%j.err" \
    call_srun_learn_job.sh
