#!/usr/bin/env bash

# Usage: bash call_sbatch.sh [--partition PARTITION] [Python options]
# --n-envs also requests n-envs + 1 CPUs; all options reach the Python job.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "This script should not be sourced, since it simply launches jobs:" >&2
    echo "--> Use:  bash ./call_sbatch.sh" >&2
    return 1
fi

#stop after any error
set -euo pipefail

DATETIME=$(date '+%Y_%m_%d_%H_%M_%S')

#proj dir is the directory where this script is located, which should be the root of the heli-gym project.
PROJ_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Only n-envs affects submission resources. Forward every Python argument intact.
# Keep the submission default (24) distinct from the standalone job default (7).
N_ENVS=24
ARGS=()
SBATCH_ARGS=()
while (($#)); do
    case "$1" in
        --n-envs)
            if (($# < 2)); then echo "Missing value for --n-envs" >&2; exit 2; fi
            N_ENVS=$2; ARGS+=("$1" "$2"); shift 2 ;;
        --n-envs=*) N_ENVS=${1#*=}; ARGS+=("$1"); shift ;;
        --partition)
            if (($# < 2)) || [[ -z "$2" || "$2" == -* ]]; then
                echo "Missing value for --partition" >&2; exit 2
            fi
            SBATCH_ARGS+=(--partition "$2"); shift 2 ;;
        --partition=*)
            [[ -n "${1#*=}" ]] || { echo "Missing value for --partition" >&2; exit 2; }
            SBATCH_ARGS+=("$1"); shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
if [[ ! "$N_ENVS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--n-envs must be a positive integer" >&2
    exit 2
fi

#PROJ_DIR="${WORKDIR:-$HOME}/SourceCode/GITHUB/heli-gym"
#PROJ_URL="https://github.com/fsperotto/heli-gym.git"
#
#if [[ ! -d "$PROJ_DIR" ]]; then
#    export HTTP_PROXY=proxy.onera:80
#    export HTTPS_PROXY=proxy.onera:80
#    mkdir -p "$(dirname "$PROJ_DIR")"
#    git clone "$PROJ_URL" "$PROJ_DIR"
#fi

LOG_DIR="$PROJ_DIR/logs/$DATETIME"
mkdir -p "$LOG_DIR"

SUBMISSION=$(sbatch --parsable \
    "${SBATCH_ARGS[@]}" \
    --job-name "heli-gym-learn-$DATETIME" \
    --chdir "$PROJ_DIR" \
    --cpus-per-task "$((N_ENVS + 1))" \
    --time 11:30:00 \
    --output "$LOG_DIR/slurm.%j.out" \
    --error "$LOG_DIR/slurm.%j.err" \
    "$PROJ_DIR/call_srun_learn_job.sh" \
        --n-envs "$N_ENVS" "${ARGS[@]}")

# --parsable returns job_id, optionally followed by ;cluster_name.
JOB_ID="${SUBMISSION%%;*}"
CLUSTER_OPTION=""
if [[ "$SUBMISSION" == *';'* ]]; then
    CLUSTER_OPTION=" --clusters=${SUBMISSION#*;}"
fi

echo
echo "Submitted job $JOB_ID ($N_ENVS environments, $((N_ENVS + 1)) CPUs)."
echo "Run these commands on $(hostname):"
echo "  Status:       squeue$CLUSTER_OPTION -j $JOB_ID"
echo "  Live status:  watch -n 5 squeue$CLUSTER_OPTION -j $JOB_ID"
echo "  Job details:  scontrol$CLUSTER_OPTION show job $JOB_ID"
echo "  Cancel:       scancel$CLUSTER_OPTION $JOB_ID"
echo "  History:      sacct$CLUSTER_OPTION -j $JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode"
echo "  Follow output: tail -f \"$LOG_DIR/slurm.$JOB_ID.out\""
echo "  Follow errors: tail -f \"$LOG_DIR/slurm.$JOB_ID.err\""
echo "Completed jobs leave squeue; use sacct to check their final state and exit code."
echo "Log files appear when the job starts. Press Ctrl+C to stop watching or following logs."
