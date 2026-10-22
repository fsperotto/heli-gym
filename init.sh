#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then

    echo "This script must be sourced so the environment remains active:" >&2
    echo "--> Use: source ./init.sh" >&2

else

    # setup_install.sh replaces this activation line with the selected environment.
    #source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/.venv/bin/activate"
    source ~/miniconda3/bin/activate

fi