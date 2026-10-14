#!/bin/sh
set -eu
SRC="${1:-/pytorch}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/gfx803-embedding-backward-row-width.patch"
TARGET="$SRC/aten/src/ATen/native/cuda/Embedding.cu"
[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi
git -C "$SRC" apply --verbose "$PATCH"
count=$(grep -c 'GFX803_EMBEDDING_BWD_ROW_WIDTH_PATCH' "$TARGET" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found in $TARGET after git apply reported success" >&2
    exit 1
fi
uses=$(grep -c 'blockDim.x\*threadIdx.y;\|blockDim.x\*blockDim.y);\|blockDim.x\*first_remaining_peer\]' "$TARGET" || true)
if [ "$uses" -ne 3 ]; then
    echo "FATAL: expected 3 row offsets using blockDim.x, found $uses" >&2
    exit 1
fi
echo "PyTorch embedding backward row-width patch applied and verified"
