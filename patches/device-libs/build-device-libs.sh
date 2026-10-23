#!/bin/sh
# Build ockl.bc from ROCm-Device-Libs with the gfx8 signal-atomic patch and stage it for the final image.
#
# HSA signals live in fine-grained host memory, and gfx8 executes a no-return atomic there as a plain store of
# its operand. ockl's signal updates discard the old value, so hostcall's doorbell never changes and device
# printf and assert() hang. The patch keeps the old value on gfx8, which selects the atomic instruction.
#
# Only ockl.bc leaves this stage. HIP code compiled in the image links it from /opt/rocm/amdgcn/bitcode;
# binaries built before the final image stage keep the copy they were linked with.
#
# Usage: build-device-libs.sh prepare | build
set -eu

SRC=/llvm-src
DL=$SRC/amd/device-libs
BUILD=/devlibs-build
OUT=/devlibs-out
PATCHES="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROCM_VERSION="${ROCM_VERSION:?ROCM_VERSION is required}"

# The commit that the release tag rocm-6.4.4 peels to, and the one the image's clang reports
# ("roc-6.4.4 25224 d366fa84..."). Device-libs bitcode must match the compiler that links it.
LLVM_COMMIT=d366fa84f3fdcbd4b10847ebd5db572ae12a34fb

case "${1:-}" in
prepare)
    [ "$ROCM_VERSION" = "6.4.4" ] || { echo "FATAL: patch is written for ROCm 6.4.4, not $ROCM_VERSION" >&2; exit 1; }
    head=$(git -C "$SRC" rev-parse HEAD)
    [ "$head" = "$LLVM_COMMIT" ] || { echo "FATAL: llvm-project is at $head, expected $LLVM_COMMIT" >&2; exit 1; }
    # A sparse checkout of one monorepo subdirectory: patch -p1, since git apply reports success there without
    # changing anything.
    patch -p1 -d "$SRC" < "$PATCHES/ockl-gfx8-signal-returning-atomics-6.4.4.patch"
    grep -q 'AF_SIGNAL(long, add,' "$DL/ockl/src/hsaqs.cl" \
        || { echo "FATAL: ockl signal patch did not apply" >&2; exit 1; }
    cmake -S "$DL" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/opt/rocm/llvm -DLLVM_DIR=/opt/rocm/llvm/lib/cmake/llvm \
        -DClang_DIR=/opt/rocm/llvm/lib/cmake/clang
    ;;
build)
    ninja -C "$BUILD" ockl
    mkdir -p "$OUT"
    cp "$BUILD/amdgcn/bitcode/ockl.bc" "$OUT/ockl.bc"
    ;;
*)
    echo "usage: $0 prepare | build" >&2; exit 1 ;;
esac
