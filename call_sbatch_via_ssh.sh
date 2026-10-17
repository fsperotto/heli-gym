#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "This script should not be sourced, since it simply launches jobs:" >&2
    echo "--> Use:  bash ./launch.sh" >&2
    return 1
fi

#USER=$(whoami)

REMOTE="juno"

#stop after any error
set -euo pipefail

#ssh fperotto@juno sbatch ~/SourceCode/GITHUB/heli_gym/call_srun_learn.sh

ssh ${USER}@${REMOTE} 'bash -s' <<'REMOTE'

DATETIME=$(date '+%Y_%m_%d_%H_%M_%S')

PROJ_DIR="${WORKDIR:-/tmp_user/juno/${USER}}/SourceCode/heli-gym"

if [[ ! -d "$PROJ_DIR" ]]; then
    export HTTP_PROXY=proxy.onera:80
    export HTTPS_PROXY=proxy.onera:80
    mkdir -p "$(dirname "$PROJ_DIR")"
    git clone https://github.com/fsperotto/heli-gym.git "$PROJ_DIR"
fi

mkdir -p "${PROJ_DIR}/logs"

sbatch \
    --chdir="$PROJ_DIR" \
    --output="$PROJ_DIR/logs/$DATETIME/slurm.%j.out" \
    --error="$PROJ_DIR/logs/$DATETIME/slurm.%j.err" \
    call_srun_learn_job.sh
REMOTE