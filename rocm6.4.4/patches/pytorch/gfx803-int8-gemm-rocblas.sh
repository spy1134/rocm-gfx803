#!/bin/sh
set -eu
SRC="${1:-/pytorch}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/gfx803-int8-gemm-rocblas.patch"
TARGET="$SRC/aten/src/ATen/cuda/CUDABlas.cpp"
[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi
git -C "$SRC" apply --verbose "$PATCH"
count=$(grep -c 'GFX803_INT8_GEMM_ROCBLAS_PATCH' "$TARGET" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found in $TARGET after git apply reported success" >&2
    exit 1
fi
grep -q 'int8_gemm_rocblas(' "$TARGET" \
    || { echo "FATAL: rocBLAS int8 path missing after git apply reported success" >&2; exit 1; }
echo "PyTorch int8_gemm rocBLAS patch applied and verified"
