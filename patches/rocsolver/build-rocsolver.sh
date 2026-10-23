#!/bin/sh
# Build rocSOLVER for one architecture against the image's patched rocBLAS and
# stage the library file for the final image.
#
# The rocSOLVER in the base image has no gfx803 code object: its fatbin holds
# gfx908 and newer only. rocSOLVER launches kernels with a bare
# hipLaunchKernelGGL and never checks hipGetLastError, so torch.linalg
# (hipSOLVER -> rocSOLVER getrf and friends) returns the unmodified input with
# stale info/ipiv and then faults, with no error at the failing call.
#
# Usage: build-rocsolver.sh configure | build
set -eu

SRC=/rocsolver-src
BUILD=/rs-build
OUT=/rs-out
ARCH="${ROCM_ARCH:?ROCM_ARCH is required}"
ROCM_VERSION="${ROCM_VERSION:?ROCM_VERSION is required}"

case "${1:-}" in
configure)
    # CMAKE_CXX_COMPILER=hipcc: the library's .cpp files carry __global__
    # kernels and take their offload flags from hip::device, so g++ fails on
    # --offload-arch. BUILD_WITH_SPARSE=OFF: only the refactoring routines
    # (csrrf) use rocSPARSE, and they are left out. Offload compression is off
    # so the fatbin stays readable by objcopy and the decode check.
    cmake -S "$SRC" -B "$BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DROCM_PATH=/opt/rocm -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DAMDGPU_TARGETS="$ARCH" \
        -DBUILD_WITH_SPARSE=OFF \
        -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF \
        -DBUILD_CLIENTS_SAMPLES=OFF \
        -DBUILD_OFFLOAD_COMPRESS=OFF -DBUILD_COMPRESSED_DBG=OFF
    ;;
build)
    jobs="${BUILD_PARALLEL_LEVEL:-auto}"
    if [ "$jobs" = "auto" ]; then
        # About 4 GB per job, and never more than 8: every rocSOLVER source file
        # instantiates its kernels for float, double and both complex types, so
        # the files are heavy, and the build host also runs a desktop.
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo)
        cpu=$(nproc)
        [ "$jobs" -gt "$cpu" ] && jobs=$cpu
        [ "$jobs" -gt 8 ] && jobs=8
        [ "$jobs" -lt 1 ] && jobs=1
    fi
    echo "rocSOLVER build: arch $ARCH, $jobs parallel jobs"
    nice -n 19 cmake --build "$BUILD" -j"$jobs"

    built="$(readlink -f "$BUILD/library/src/librocsolver.so")"
    [ -f "$built" ] || { echo "FATAL: no librocsolver.so under $BUILD/library/src" >&2; exit 1; }

    # The base image's librocsolver.so.0 symlink points at
    # librocsolver.so.0.4.<major*10000+minor*100+patch of ROCm>. The file must
    # land under that exact name, or the copy sits unused beside the stock one.
    suffix=$(echo "$ROCM_VERSION" | awk -F. '{printf "%d%02d%02d", $1, $2, $3}')
    name="librocsolver.so.0.4.$suffix"
    mkdir -p "$OUT"
    cp "$built" "$OUT/$name"

    soname=$(readelf -d "$OUT/$name" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')
    [ "$soname" = "librocsolver.so.0" ] || {
        echo "FATAL: SONAME is '$soname', hipSOLVER needs librocsolver.so.0" >&2; exit 1; }

    objcopy -O binary --only-section=.hip_fatbin "$OUT/$name" /tmp/fatbin.bin
    size=$(stat -c%s /tmp/fatbin.bin)
    # A host-only stub library is the failure mode being fixed; a real gfx803
    # build holds megabytes of code objects.
    [ "$size" -ge 1000000 ] || {
        echo "FATAL: .hip_fatbin is $size bytes, too small for real $ARCH device code" >&2; exit 1; }
    grep -aq "amdgcn-amd-amdhsa--$ARCH" /tmp/fatbin.bin || {
        echo "FATAL: no $ARCH bundle entry in the fatbin" >&2; exit 1; }
    rm -f /tmp/fatbin.bin
    echo "OK: $name is a $ARCH build ($size byte .hip_fatbin)"
    ;;
*)
    echo "usage: $0 configure|build" >&2; exit 2
    ;;
esac
