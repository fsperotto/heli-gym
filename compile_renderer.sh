#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RENDERER_SOURCE="$SCRIPT_DIR/heligym/envs/renderer"
RENDERER_BUILD="$RENDERER_SOURCE/build"
RENDERER_BUILD_TYPE="${RENDERER_BUILD_TYPE:-Release}"

if ! command -v cmake >/dev/null 2>&1; then
    echo "CMake is required to compile the renderer." >&2
    exit 1
fi

echo "Configuring $RENDERER_BUILD_TYPE renderer build..."
cmake \
    -S "$RENDERER_SOURCE" \
    -B "$RENDERER_BUILD" \
    -DCMAKE_BUILD_TYPE="$RENDERER_BUILD_TYPE"

echo "Compiling renderer..."
cmake --build "$RENDERER_BUILD" --config "$RENDERER_BUILD_TYPE" --parallel

echo "Renderer compiled successfully."
echo "Native library output: $RENDERER_SOURCE/bin"
