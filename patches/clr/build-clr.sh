#!/bin/sh
# Build libamdhip64 from ROCm/clr with the gfx8 cache-maintenance patch and stage the library file for the
# final image.
#
# The libamdhip64 in the base image never asks the command processor to invalidate the per-CU vector L1 on
# gfx803. The firmware does not act on the AQL acquire fence scope either, so a kernel can read the old
# contents of a page that a copy kernel has just refilled (hipFree followed by hipMalloc of a small buffer
# hands the same page back). The patch publishes a PM4 ACQUIRE_MEM in the ring ahead of every dispatch.
#
# Only libamdhip64 leaves this stage. The patch touches nothing in ROCR-Runtime, libhiprtc or any other
# library.
#
# Usage: build-clr.sh prepare | build
set -eu

CLR=/clr-src
HIP=/hip-src
BUILD=/clr-build
OUT=/clr-out
PATCHES="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROCM_VERSION="${ROCM_VERSION:?ROCM_VERSION is required}"

# Commits that the release tag rocm-6.4.4 peels to. A tag that was moved would otherwise build silently
# against different sources than the patch was written for.
CLR_COMMIT=123eb5128769ba30fda8f84414421bc858e8c695

case "${1:-}" in
prepare)
    [ "$ROCM_VERSION" = "6.4.4" ] || { echo "FATAL: patch is written for ROCm 6.4.4, not $ROCM_VERSION" >&2; exit 1; }
    head=$(git -C "$CLR" rev-parse HEAD)
    [ "$head" = "$CLR_COMMIT" ] || { echo "FATAL: clr is at $head, expected $CLR_COMMIT" >&2; exit 1; }
    sh "$PATCHES/clr-gfx8-tc-invalidate-6.4.4.sh" "$CLR"
    # HIP_COMMON_DIR/VERSION decides the library version string.
    [ -f "$HIP/VERSION" ] || { echo "FATAL: no VERSION file in $HIP" >&2; exit 1; }
    # The OpenCL runtime is not part of the image.
    cmake -S "$CLR" -B "$BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
        -DHIP_COMMON_DIR="$HIP" -DHIPCC_BIN_DIR=/opt/rocm/bin \
        -DROCM_PATH=/opt/rocm -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_INSTALL_PREFIX=/clr-install \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    ;;
build)
    jobs="${BUILD_PARALLEL_LEVEL:-auto}"
    if [ "$jobs" = "auto" ]; then
        jobs=$(nproc)
    fi
    # The machine that builds this also runs other work.
    [ "$jobs" -gt 6 ] && jobs=6
    echo "CLR build: $jobs parallel jobs"
    nice -n 19 cmake --build "$BUILD" --target amdhip64 -j"$jobs"

    built="$(find "$BUILD" -name 'libamdhip64.so.*' -type f | head -1)"
    [ -f "$built" ] || { echo "FATAL: no libamdhip64.so.* under $BUILD" >&2; exit 1; }

    # The base image's libamdhip64.so.6 symlink points at libamdhip64.so.6.4.<ROCm version number>; the file
    # must land under that name or the copy sits unused beside the stock one.
    suffix=$(echo "$ROCM_VERSION" | awk -F. '{printf "%d%02d%02d", $1, $2, $3}')
    name="libamdhip64.so.6.4.$suffix"
    stock="/opt/rocm/lib/$name"
    [ -f "$stock" ] || { echo "FATAL: the base image has no $stock to replace" >&2; exit 1; }
    mkdir -p "$OUT"
    cp "$built" "$OUT/$name"
    strip --strip-debug "$OUT/$name"

    soname=$(readelf -d "$OUT/$name" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')
    [ "$soname" = "libamdhip64.so.6" ] || { echo "FATAL: SONAME is '$soname', dependants need libamdhip64.so.6" >&2; exit 1; }

    # Same dependencies as the stock library, otherwise the final image may lack one of them.
    needed() { readelf -d "$1" | sed -n 's/.*NEEDED.*\[\(.*\)\]/\1/p' | sort; }
    if [ "$(needed "$OUT/$name")" != "$(needed "$stock")" ]; then
        echo "FATAL: NEEDED libraries differ from the stock library:" >&2
        needed "$stock" > /tmp/needed.stock
        needed "$OUT/$name" > /tmp/needed.new
        diff /tmp/needed.stock /tmp/needed.new >&2 || true
        exit 1
    fi

    # Marker strings the patch adds; absence means the object was built from unpatched sources.
    grep -aq 'CLR_GFX8_TC_INVALIDATE' "$OUT/$name" || { echo "FATAL: CLR_GFX8_TC_INVALIDATE is not in $name" >&2; exit 1; }
    grep -aq 'CLR_GFX8_TC_RECORD_FENCE' "$OUT/$name" || { echo "FATAL: CLR_GFX8_TC_RECORD_FENCE is not in $name" >&2; exit 1; }

    echo "OK: $name ($(stat -c%s "$OUT/$name") bytes, stock $(stat -c%s "$stock"))"
    ;;
*)
    echo "usage: $0 prepare|build" >&2; exit 2
    ;;
esac
