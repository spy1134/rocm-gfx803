#!/bin/sh
# Apply the gfx8 code generation patches (and the ockl signal patch, so libamd_comgr embeds the same ockl.bc as the
# image) to a ROCm/llvm-project rocm-6.4.4 checkout and make sure that each apply worked. The patch files give the
# reason and the change.
#
# Uses patch -p1: the checkout is sparse, and git apply reports success on a sparse checkout without changing anything.
#
# Usage: llvm-gfx8-codegen-6.4.4.sh <llvm-project dir> <dir holding ockl-gfx8-signal-returning-atomics-6.4.4.patch>
set -eu

SRC="${1:-/llvm-src}"
DEVLIBS_PATCHES="${2:-/devlibs-patches}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

ISEL="$SRC/llvm/lib/Target/AMDGPU/SIISelLowering.cpp"
DAG="$SRC/llvm/lib/Target/AMDGPU/AMDGPUISelDAGToDAG.cpp"
FIXSGPR="$SRC/llvm/lib/Target/AMDGPU/SIFixSGPRCopies.cpp"
HSAQS="$SRC/amd/device-libs/ockl/src/hsaqs.cl"

for f in "$ISEL" "$DAG" "$FIXSGPR" "$HSAQS"; do
    [ -f "$f" ] || { echo "FATAL: $f is missing; the sparse checkout lacks a directory the patches touch" >&2; exit 1; }
done

apply() {
    patch -p1 -d "$SRC" < "$1"
}

apply "$DEVLIBS_PATCHES/ockl-gfx8-signal-returning-atomics-6.4.4.patch"
apply "$SELF_DIR/0001-amdgpu-callee-implicit-kernarg.patch"
apply "$SELF_DIR/0002-amdgpu-gfx8-flat-lds-m0.patch"

grep -q 'AF_SIGNAL(long, add,' "$HSAQS" \
    || { echo "FATAL: ockl signal patch did not apply" >&2; exit 1; }
grep -q 'A callee has no kernarg segment pointer' "$ISEL" \
    || { echo "FATAL: callee implicit-kernarg patch did not apply to $ISEL" >&2; exit 1; }
grep -q 'A generic pointer can hit the LDS aperture' "$DAG" \
    || { echo "FATAL: flat-LDS M0 patch (selection hunk) did not apply to $DAG" >&2; exit 1; }
grep -q 'range-check LDS accesses against M0' "$FIXSGPR" \
    || { echo "FATAL: flat-LDS M0 patch (SIFixSGPRCopies hunk) did not apply to $FIXSGPR" >&2; exit 1; }
echo "OK: ockl signal, callee implicit-kernarg and flat-LDS M0 patches applied"
