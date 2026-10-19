#!/usr/bin/env bash

ask_yes_no() {
    local prompt="$1"
    local reply

    while true; do
        if ! read -r -p "$prompt [y/n] " reply; then
            echo "Input closed; installation cancelled." >&2
            return 1
        fi
        case "$reply" in
            [Yy]|[Yy][Ee][Ss]) return 0 ;;
            [Nn]|[Nn][Oo]) return 1 ;;
            *) echo "Please answer yes or no." ;;
        esac
    done
}

# A subshell keeps sourcing this script from changing the caller's directory
# or shell options. Activate .venv explicitly after installation if needed.
install_heligym() (
    set -euo pipefail
    PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    VENV_DIR="$PROJECT_DIR/.venv"
    cd -- "$PROJECT_DIR"

    PACKAGE_MANAGER=uv
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is not installed."
        if ask_yes_no "Would you like to install uv?"; then
            if ! command -v curl >/dev/null 2>&1; then
                echo "Cannot install uv automatically because curl is not installed." >&2
                return 1
            fi
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
            if ! command -v uv >/dev/null 2>&1; then
                echo "uv is unavailable on PATH after installation." >&2
                return 1
            fi
        else
            PACKAGE_MANAGER=pip
            echo "Using pip instead."
        fi
    fi

    PYTHON_BIN=""
    USE_ACTIVE_PYTHON=false
    if [ -e "$VENV_DIR" ]; then
        if ! ask_yes_no "A virtual environment already exists at $VENV_DIR. Would you like to use it?"; then
            echo "Installation cancelled. Please remove $VENV_DIR and rerun this script to reinstall." >&2
            return 0
        fi
        if [ ! -x "$VENV_DIR/bin/python" ] || [ ! -f "$VENV_DIR/pyvenv.cfg" ]; then
            echo "$VENV_DIR does not appear to be a valid virtual environment." >&2
            return 1
        fi
        PYTHON_BIN="$VENV_DIR/bin/python"
    else
        for candidate in python python3; do
            if command -v "$candidate" >/dev/null 2>&1 &&
               ACTIVE_PYTHON=$("$candidate" -c 'import sys; print(sys.executable)'); then
                if ask_yes_no "Use the currently available Python ($ACTIVE_PYTHON) and install packages into its environment?"; then
                    PYTHON_BIN="$ACTIVE_PYTHON"
                    USE_ACTIVE_PYTHON=true
                fi
                break
            fi
        done
        if [ -z "$PYTHON_BIN" ]; then
            if ! ask_yes_no "No virtual environment exists at $VENV_DIR. Would you like to create it?"; then
                echo "Installation cancelled."
                return 0
            fi
            if [ "$PACKAGE_MANAGER" = uv ]; then
                PYTHON_ARGS=(--python 3.12)
                if ! uv python find 3.12 --no-python-downloads >/dev/null 2>&1; then
                    if ask_yes_no "Python 3.12 was not found. Would you like uv to install it?"; then
                        uv python install 3.12
                    elif ask_yes_no "Would you like to use another installed Python compatible with pyproject.toml?"; then
                        PYTHON_ARGS=()
                    else
                        echo "Installation cancelled."
                        return 0
                    fi
                fi
                uv venv "$VENV_DIR" "${PYTHON_ARGS[@]}" --no-python-downloads
            else
                if [ -z "${ACTIVE_PYTHON:-}" ]; then
                    echo "Cannot create .venv: install Python with venv support first." >&2
                    return 1
                fi
                "$ACTIVE_PYTHON" -m venv "$VENV_DIR"
            fi
            PYTHON_BIN="$VENV_DIR/bin/python"
        fi
    fi

    if [ "$PACKAGE_MANAGER" = uv ]; then
        INSTALL_ARGS=()
        if [ "$USE_ACTIVE_PYTHON" = true ]; then
            INSTALL_ARGS+=(--system)
        fi
        uv pip install -e '.[train]' --python "$PYTHON_BIN" --no-python-downloads "${INSTALL_ARGS[@]}"
    else
        if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
            echo "pip is unavailable for $PYTHON_BIN. Install pip for this interpreter and rerun this script." >&2
            return 1
        fi
        "$PYTHON_BIN" -m pip install -e '.[train]'
    fi
    echo "Building the native renderer..."
    if ! bash "$PROJECT_DIR/compile_renderer.sh"; then
        echo "Native renderer build failed." >&2
        echo "On Ubuntu/Debian, install the build dependencies with:" >&2
        echo "  sudo apt install build-essential cmake libglfw3-dev libassimp-dev libgl1-mesa-dev" >&2
        return 1
    fi
    if [ "$USE_ACTIVE_PYTHON" = true ]; then
        printf 'Installation complete using %s.\n' "$PYTHON_BIN"
    else
        printf 'Installation complete. Activate the environment with: source "%s/bin/activate"\n' "$VENV_DIR"
    fi
)

install_heligym
