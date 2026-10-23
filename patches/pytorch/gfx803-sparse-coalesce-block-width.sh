#!/bin/sh
set -eu
SRC="${1:-/pytorch}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/gfx803-sparse-coalesce-block-width.patch"
TARGET="$SRC/aten/src/ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh"
[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi
git -C "$SRC" apply --verbose "$PATCH"
count=$(grep -c 'GFX803_SPARSE_COALESCE_BLOCK_WIDTH' "$TARGET" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found in $TARGET after git apply reported success" >&2
    exit 1
fi
uses=$(grep -c 'startFeature + ii \* blockDim.x' "$TARGET" || true)
if [ "$uses" -ne 4 ]; then
    echo "FATAL: expected 4 feature strides using blockDim.x, found $uses" >&2
    exit 1
fi
if grep -q 'startFeature + ii \* C10_WARP_SIZE' "$TARGET"; then
    echo "FATAL: a feature stride still uses C10_WARP_SIZE in $TARGET" >&2
    exit 1
fi
echo "PyTorch sparse coalesce block-width patch applied and verified"
