#!/bin/sh
set -eu
SRC="${1:-/pytorch}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/gfx803-logical-warp-smem.patch"
CUDA_DIR="$SRC/aten/src/ATen/native/cuda"
[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi
git -C "$SRC" apply --verbose "$PATCH"
count=$(grep -c 'GFX803_LOGICAL_WARP_SMEM' "$CUDA_DIR/block_reduce.cuh" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found in block_reduce.cuh after git apply reported success" >&2
    exit 1
fi
check() {
    file="$1"; want="$2"
    got=$(grep -c 'device_warp_size()' "$CUDA_DIR/$file" || true)
    if [ "$got" -ne "$want" ]; then
        echo "FATAL: expected $want uses of device_warp_size() in $file, found $got" >&2
        exit 1
    fi
}
check SoftMax.cu 5
check Embedding.cu 1
check MultinomialKernel.cu 1
check DepthwiseConv2d.cu 1
check PersistentSoftmax.cuh 2
echo "PyTorch logical-warp shared-memory patch applied and verified"
