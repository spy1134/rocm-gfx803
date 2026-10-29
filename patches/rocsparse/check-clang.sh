#!/bin/sh
# Compile a saturating unsigned subtract of a 32-bit literal for gfx803 with the installed clang and make sure the
# result keeps its clamp. The compiler without 0004-amdgpu-gfx8-carry-add-sub-clamp.patch folds the literal into
# v_subrev_u32_e32, which has no clamp bit, and `x < 1000 ? 0 : x - 1000` wraps for x < 1000. A stage that compiles
# device code with the patched clang-19 runs this first, so a stale or missing copy stops the build.
#
# Needs no GPU. Usage: check-clang.sh [clang++ to test]
set -eu

CXX="${1:-/opt/rocm/llvm/bin/clang++}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/k.hip" <<'HIP'
#include <hip/hip_runtime.h>
__global__ void k(unsigned *p) { unsigned x = p[threadIdx.x]; p[threadIdx.x] = x < 1000u ? 0u : x - 1000u; }
HIP

"$CXX" -x hip --offload-arch=gfx803 --offload-device-only -O3 -S "$WORK/k.hip" -o "$WORK/k.s"
if grep -q 'v_subrev_u32_e32 v[0-9]*, vcc, 0x3e8' "$WORK/k.s"; then
    echo "FAIL: $CXX drops the clamp of a saturating subtract of a literal on gfx803" >&2
    exit 1
fi
grep -q 'v_sub_u32_e64 .* clamp' "$WORK/k.s" \
    || { echo "FATAL: no clamped subtract in the probe's code; the probe no longer tests the fold" >&2; exit 1; }
echo "OK: $CXX keeps the clamp of a saturating subtract of a literal on gfx803"
