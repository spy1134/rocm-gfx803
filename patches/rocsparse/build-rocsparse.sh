#!/bin/sh
# Build rocSPARSE for one architecture against the image's ROCm and stage the library file for the final image.
#
# The rocSPARSE in the base image carries device code for gfx908 and newer only. A launch for an architecture
# missing from the library does nothing and reports no error, so torch.sparse.mm and CSR products on the GPU
# return wrong values without any message.
#
# Usage: build-rocsparse.sh configure | build
set -eu

SRC=/rocsparse-src
BUILD=/rsp-build
OUT=/rsp-out
ARCH="${ROCM_ARCH:?ROCM_ARCH is required}"
ROCM_VERSION="${ROCM_VERSION:?ROCM_VERSION is required}"

case "${1:-}" in
configure)
    # hipcc as the C++ compiler: the library sources carry __global__ kernels and g++ rejects --offload-arch.
    cmake -S "$SRC" -B "$BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DROCM_PATH=/opt/rocm -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DAMDGPU_TARGETS="$ARCH" \
        -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF -DBUILD_CLIENTS_SAMPLES=OFF \
        -DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=OFF
    ;;
build)
    jobs="${BUILD_PARALLEL_LEVEL:-auto}"
    if [ "$jobs" = "auto" ]; then
        # Every source file instantiates its kernels for several index and value types; about 4 GB per job.
        jobs=$(awk '/MemAvailable/{printf "%d", $2/1024/1024/4}' /proc/meminfo)
        cpu=$(nproc)
        [ "$jobs" -gt "$cpu" ] && jobs=$cpu
        [ "$jobs" -gt 8 ] && jobs=8
        [ "$jobs" -lt 1 ] && jobs=1
    fi
    echo "rocSPARSE build: arch $ARCH, $jobs parallel jobs"
    nice -n 19 cmake --build "$BUILD" -j"$jobs"

    built="$(readlink -f "$BUILD/library/librocsparse.so")"
    [ -f "$built" ] || { echo "FATAL: no librocsparse.so under $BUILD/library" >&2; exit 1; }

    # The base image's librocsparse.so.1 symlink points at librocsparse.so.1.0.<ROCm version number>; the file must
    # land under that name or the copy sits unused beside the stock one.
    suffix=$(echo "$ROCM_VERSION" | awk -F. '{printf "%d%02d%02d", $1, $2, $3}')
    name="librocsparse.so.1.0.$suffix"
    mkdir -p "$OUT"
    cp "$built" "$OUT/$name"

    soname=$(readelf -d "$OUT/$name" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')
    [ "$soname" = "librocsparse.so.1" ] || { echo "FATAL: SONAME is '$soname', dependants need librocsparse.so.1" >&2; exit 1; }

    objcopy -O binary --only-section=.hip_fatbin "$OUT/$name" /tmp/fatbin.bin
    size=$(stat -c%s /tmp/fatbin.bin)
    [ "$size" -ge 1000000 ] || { echo "FATAL: .hip_fatbin is $size bytes, too small for real $ARCH device code" >&2; exit 1; }
    # The fatbin is zstd-compressed (CCOB): the target ids sit in the compressed header of each bundle, so decode
    # the first bytes of every bundle with the zstd tool and look for the architecture there.
    if ! grep -aq "amdgcn-amd-amdhsa--$ARCH" /tmp/fatbin.bin; then
        python3 - "$ARCH" <<'PY' || { echo "FATAL: no $ARCH bundle entry in the fatbin" >&2; exit 1; }
import re, subprocess, sys
arch = sys.argv[1].encode()
d = open("/tmp/fatbin.bin", "rb").read()
found = 0
for m in re.finditer(rb"CCOB\x02\x00", d):
    z = d.find(b"\x28\xb5\x2f\xfd", m.start(), m.start() + 64)
    if z < 0:
        continue
    head = subprocess.run(["zstd", "-dc", "--no-check"], input=d[z:z + 16384], capture_output=True).stdout[:8192]
    found += head.count(b"amdgcn-amd-amdhsa--" + arch)
sys.exit(0 if found else 1)
PY
    fi
    rm -f /tmp/fatbin.bin
    echo "OK: $name is a $ARCH build ($size byte .hip_fatbin)"
    ;;
*)
    echo "usage: $0 configure|build" >&2; exit 2
    ;;
esac
