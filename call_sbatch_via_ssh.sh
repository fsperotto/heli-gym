#!/usr/bin/env bash

# Usage: bash call_sbatch_via_ssh.sh [--user USER] [--remote HOST] [Python options]
# Options accept --name value or --name=value. Only user/remote are consumed here.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "This script should not be sourced, since it simply launches jobs:" >&2
    echo "--> Use:  bash ./call_sbatch_via_ssh.sh" >&2
    return 1
fi

set -euo pipefail

SSH_USER=${USER:-$(whoami)}
REMOTE="juno"
FORWARD_ARGS=()
while (($#)); do
    case "$1" in
        --user|--remote)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            if [[ "$1" == --user ]]; then SSH_USER=$2; else REMOTE=$2; fi
            shift 2 ;;
        --user=*) SSH_USER="${1#*=}"; shift ;;
        --remote=*) REMOTE="${1#*=}"; shift ;;
        *) FORWARD_ARGS+=("$1"); shift ;;
    esac
done
[[ -n "$SSH_USER" && -n "$REMOTE" ]] || { echo "User and remote must not be empty" >&2; exit 2; }

#get current script directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Stream the shared launcher so it also works before the remote clone exists.
# Use /dev/stdin so the launcher's check against sourcing remains effective.
# SSH joins command arguments into shell text; quote each argument for remote Bash.
printf -v REMOTE_COMMAND 'bash /dev/stdin'
if ((${#FORWARD_ARGS[@]})); then
    printf -v QUOTED_ARGS ' %q' "${FORWARD_ARGS[@]}"
    REMOTE_COMMAND+="$QUOTED_ARGS"
fi
ssh -- "${SSH_USER}@${REMOTE}" "$REMOTE_COMMAND" < "${SCRIPT_DIR}/call_sbatch.sh"
