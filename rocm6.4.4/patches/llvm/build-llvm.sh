#!/bin/sh
# Build clang-19 and libamd_comgr from ROCm/llvm-project with the gfx8 code generation patches and stage the two
# files for the PyTorch build and the final image.
#
# The compiler in the base image reads a non-inlined function's implicit kernel arguments (queue pointer for s_trap,
# private and shared aperture bases) from address 0 on gfx8, and leaves M0 unset for generic-pointer accesses that
# land in LDS. libamd_comgr embeds its own copy of the compiler, so hipRTC, MIOpen's run-time kernels, MIGraphX's
# JIT and PyTorch's jiterator carry the same code generation as hipcc and need the same replacement; its embedded
# ockl.bc also comes from the tree built here, so it carries the signal patch that the image's ockl.bc has.
#
# Only clang-19 and libamd_comgr leave this stage. lld, llc, opt and clang-linker-wrapper embed the same code
# generator but are only reached through -fgpu-rdc and LTO links, which nothing in this image builds with, and
# the other tools in /opt/rocm/lib/llvm/bin stay as shipped.
#
# Usage: build-llvm.sh prepare | build
set -eu

SRC=/llvm-src
BUILD=/llvm-build
OUT=/llvm-out
PATCHES="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROCM_VERSION="${ROCM_VERSION:?ROCM_VERSION is required}"

# The commit that the release tag rocm-6.4.4 peels to, and the one the shipped compiler reports.
LLVM_COMMIT=d366fa84f3fdcbd4b10847ebd5db572ae12a34fb

case "${1:-}" in
prepare)
    [ "$ROCM_VERSION" = "6.4.4" ] || { echo "FATAL: patches are written for ROCm 6.4.4, not $ROCM_VERSION" >&2; exit 1; }
    head=$(git -C "$SRC" rev-parse HEAD)
    [ "$head" = "$LLVM_COMMIT" ] || { echo "FATAL: llvm-project is at $head, expected $LLVM_COMMIT" >&2; exit 1; }
    sh "$PATCHES/llvm-gfx8-codegen-6.4.4.sh" "$SRC" /devlibs-patches
    ;;
build)
    jobs="${BUILD_PARALLEL_LEVEL:-auto}"
    if [ "$jobs" = "auto" ]; then
        jobs=$(nproc)
    fi
    # The machine that builds this also runs other work, and the large compile jobs and the final links are memory-heavy.
    [ "$jobs" -gt 12 ] && jobs=12
    echo "LLVM build: $jobs parallel jobs"

    # Same targets as the shipped compiler (clang -print-targets: amdgcn, r600, x86). The vendor and repository
    # strings reproduce `clang --version` of the shipped binary, which tools and caches key on. COMGR_DISABLE_SPIRV:
    # the SPIR-V translator is not part of llvm-project and only amdgcnspirv targets use it.
    configure() {
        cmake -S "$SRC/llvm" -B "$BUILD" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF \
            -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
            -DLLVM_EXTERNAL_PROJECTS="$1" \
            -DLLVM_EXTERNAL_DEVICE_LIBS_SOURCE_DIR="$SRC/amd/device-libs" \
            -DLLVM_EXTERNAL_COMGR_SOURCE_DIR="$SRC/amd/comgr" \
            -DCOMGR_DISABLE_SPIRV=ON -DBUILD_TESTING=OFF \
            -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
            -DCLANG_VENDOR="AMD " \
            -DLLVM_FORCE_VC_REPOSITORY="https://github.com/RadeonOpenCompute/llvm-project roc-6.4.4 25224" \
            -DLLVM_FORCE_VC_REVISION="$LLVM_COMMIT" \
            -DLLVM_ENABLE_ZLIB=FORCE_ON -DLLVM_ENABLE_ZSTD=FORCE_ON \
            -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF -DLLVM_ENABLE_TERMINFO=OFF \
            -DLLVM_PARALLEL_LINK_JOBS=2 \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    }
    # comgr's configure step hashes opencl-c.h from the build tree's resource directory, so the headers have to be
    # generated before comgr is added to the build.
    configure device-libs
    nice -n 19 ninja -C "$BUILD" -j"$jobs" clang-resource-headers
    configure "device-libs;comgr"
    nice -n 19 ninja -C "$BUILD" -j"$jobs" clang amd_comgr

    mkdir -p "$OUT"
    suffix=$(echo "$ROCM_VERSION" | awk -F. '{printf "%d%02d%02d", $1, $2, $3}')
    comgr="libamd_comgr.so.3.0.$suffix"
    cp "$BUILD/bin/clang-19" "$OUT/clang-19"
    cp "$(readlink -f "$BUILD/lib/libamd_comgr.so.3")" "$OUT/$comgr"
    strip --strip-debug "$OUT/$comgr"

    stock_clang=/opt/rocm/lib/llvm/bin/clang-19
    stock_comgr="/opt/rocm/lib/$comgr"
    [ -f "$stock_clang" ] && [ -f "$stock_comgr" ] \
        || { echo "FATAL: the base image has no $stock_clang or $stock_comgr to replace" >&2; exit 1; }

    # The same version line as AMD's build, otherwise something that parses it treats the compiler as foreign.
    want=$("$stock_clang" --version | sed -n 1p)
    got=$("$OUT/clang-19" --version | sed -n 1p)
    [ "$got" = "$want" ] || { echo "FATAL: clang reports '$got', the shipped one '$want'" >&2; exit 1; }
    # Same code generators registered.
    [ "$("$OUT/clang-19" -print-targets)" = "$("$stock_clang" -print-targets)" ] \
        || { echo "FATAL: registered targets differ from the shipped clang-19" >&2; exit 1; }

    # Dependants load libamd_comgr by soname and call it through the public C API: the library has to export the
    # same versioned symbols and need the same shared libraries.
    needed() { readelf -d "$1" | sed -n 's/.*NEEDED.*\[\(.*\)\]/\1/p' | sort; }
    exports() { nm -D --defined-only "$1" | awk '{print $2, $3}' | sort; }
    soname=$(readelf -d "$OUT/$comgr" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')
    [ "$soname" = "libamd_comgr.so.3" ] || { echo "FATAL: SONAME is '$soname'" >&2; exit 1; }
    [ "$(needed "$OUT/$comgr")" = "$(needed "$stock_comgr")" ] || { echo "FATAL: NEEDED libraries differ from the shipped libamd_comgr" >&2; exit 1; }
    [ "$(exports "$OUT/$comgr")" = "$(exports "$stock_comgr")" ] || { echo "FATAL: exported symbols differ from the shipped libamd_comgr" >&2; exit 1; }

    # The aperture load in a callee has to come from the implicit-argument pointer in s[8:9]. The shipped compiler
    # must still show the null-base load on the same source, otherwise this check proves nothing.
    probe=/tmp/llvm-probe
    mkdir -p "$probe"
    cat > "$probe/callee.hip" <<'HIP'
#include <hip/hip_runtime.h>
__device__ int *g;
__attribute__((noinline)) __device__ void callee(int i) { __shared__ int a[64]; g = a + i; }
__global__ void k(int *o) { callee(threadIdx.x); *g = 1; o[threadIdx.x] = 0; }
HIP
    res=/opt/rocm/lib/llvm/lib/clang/19
    "$stock_clang" -x hip --rocm-path=/opt/rocm -resource-dir="$res" --offload-arch=gfx803 --offload-device-only \
        -O2 -S "$probe/callee.hip" -o "$probe/stock.s"
    "$OUT/clang-19" -x hip --rocm-path=/opt/rocm -resource-dir="$res" --offload-arch=gfx803 --offload-device-only \
        -O2 -S "$probe/callee.hip" -o "$probe/new.s"
    grep -q 's_mov_b64 s\[[0-9:]*\], 0xc4$' "$probe/stock.s" \
        || { echo "FATAL: the shipped compiler no longer shows the null-base aperture load; the probe is stale" >&2; exit 1; }
    grep -q 's_load_dword s[0-9]*, s\[8:9\], 0xc4' "$probe/new.s" \
        || { echo "FATAL: the built compiler does not load the shared aperture through s[8:9]" >&2; exit 1; }
    ! grep -q 's_mov_b64 s\[[0-9:]*\], 0xc4$' "$probe/new.s" \
        || { echo "FATAL: the built compiler still loads the shared aperture from a null base" >&2; exit 1; }

    echo "OK: clang-19 ($(stat -c%s "$OUT/clang-19") bytes, shipped $(stat -c%s "$stock_clang")), $comgr ($(stat -c%s "$OUT/$comgr") bytes, shipped $(stat -c%s "$stock_comgr"))"
    ;;
*)
    echo "usage: $0 prepare|build" >&2; exit 2
    ;;
esac
