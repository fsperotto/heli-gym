#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "This script should not be sourced, since it simply launches jobs:" >&2
    echo "--> Use:  bash ./call_sbatch_via_ssh.sh" >&2
    return 1
fi

USER=${USER:-$(whoami)}
REMOTE="juno"

#get current script directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Stream the shared launcher so it also works before the remote clone exists.
# Use /dev/stdin so the launcher's check against sourcing remains effective.
ssh "${USER}@${REMOTE} bash /dev/stdin" < "${SCRIPT_DIR}/call_sbatch.sh"