#!/bin/sh
set -eu
SRC="${1:-/pytorch}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/gfx803-layer-norm-warp-size.patch"
TARGET="$SRC/aten/src/ATen/native/cuda/layer_norm_kernel.cu"
[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi
git -C "$SRC" apply --verbose "$PATCH"
count=$(grep -c 'GFX803_LAYER_NORM_WARP_SIZE_PATCH' "$TARGET" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found in $TARGET after git apply reported success" >&2
    exit 1
fi
uses=$(grep -c 'const int warp_size = layer_norm_warp_size();' "$TARGET" || true)
if [ "$uses" -ne 2 ]; then
    echo "FATAL: expected 2 launchers using layer_norm_warp_size(), found $uses" >&2
    exit 1
fi
echo "PyTorch layer_norm warp-size patch applied and verified"
