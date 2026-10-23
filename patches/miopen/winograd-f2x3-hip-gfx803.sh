#!/bin/sh
set -eu
SRC="${1:-/miopen-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/winograd-f2x3-hip-gfx803.patch"
[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi
git -C "$SRC" apply --verbose "$PATCH"
for f in src/solver.cpp src/mlo_dir_conv.cpp src/include/miopen/conv/solvers.hpp; do
    count=$(grep -c 'conv::ConvHipWinogradF2x3\|struct ConvHipWinogradF2x3 ' "$SRC/$f" || true)
    if [ "$count" -ne 1 ]; then
        echo "FATAL: ConvHipWinogradF2x3 not registered in $f after git apply reported success" >&2
        exit 1
    fi
done
for f in src/solver/conv/conv_hip_winograd_f2x3.cpp \
         src/kernels/gpu_winograd_f2x3_kernel/winograd_f2x3_xform.cpp; do
    [ -f "$SRC/$f" ] || { echo "FATAL: $f missing after git apply reported success" >&2; exit 1; }
done
grep -q 'miopenWinogradF2x3XformInHalf' "$SRC/src/kernels/gpu_winograd_f2x3_kernel/winograd_f2x3_xform.cpp" \
    || { echo "FATAL: fp16 transform kernels missing after git apply reported success" >&2; exit 1; }
grep -q 'GPU_WINOGRAD_F2X3_KERNEL_HIP' "$SRC/src/CMakeLists.txt" \
    || { echo "FATAL: transform kernels not added to the kernel list" >&2; exit 1; }
echo "Winograd F(2x2,3x3) HIP solver patch applied and verified"
