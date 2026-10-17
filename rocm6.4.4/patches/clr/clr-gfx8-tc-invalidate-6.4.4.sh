#!/bin/sh
# Apply clr-gfx8-tc-invalidate-6.4.4.patch to a ROCm/clr rocm-6.4.4 checkout and make sure
# that the apply worked. The patch file gives the reason and the change.
#
# On gfx7 and gfx8 the CP does not act on the AQL acquire and release fence scopes, so ROCclr
# publishes one PM4 ACQUIRE_MEM ring slot (vector L1 and scalar cache invalidate) ahead of every
# kernel dispatch, inside releaseGpuMemoryFence() before copy-engine work, and ahead of an
# event-record barrier.
#
# Uses patch -p1: the 6.4.4 line builds CLR from a tarball, not a git checkout.
set -eu

SRC="${1:-/clr-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/clr-gfx8-tc-invalidate-6.4.4.patch"
CPP="$SRC/rocclr/device/rocm/rocvirtual.cpp"
HPP="$SRC/rocclr/device/rocm/rocvirtual.hpp"
FLAGS="$SRC/rocclr/utils/flags.hpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -f "$CPP" ] && [ -f "$HPP" ] && [ -f "$FLAGS" ] || {
    echo "FATAL: rocclr sources missing under $SRC" >&2; exit 1; }

if grep -q 'insertTcFence' "$CPP"; then
    echo "already patched, skipping"
    exit 0
fi

patch -p1 -d "$SRC" < "$PATCH"

[ "$(grep -c 'if (!insertTcFence()) return false;' "$CPP")" -eq 2 ] || {
    echo "FATAL: expected 2 returning insertTcFence() call sites in $CPP, found $(
        grep -c 'if (!insertTcFence()) return false;' "$CPP")" >&2; exit 1; }
grep -q 'CLR_GFX8_TC_RECORD_FENCE && !insertTcFence()' "$CPP" || {
    echo "FATAL: event-record TC writeback site missing from $CPP" >&2; exit 1; }
grep -q 'bool VirtualGPU::insertTcFence()' "$CPP" || {
    echo "FATAL: insertTcFence() definition missing from $CPP" >&2; exit 1; }
grep -q 'bool insertTcFence();' "$HPP" || {
    echo "FATAL: insertTcFence() declaration missing from $HPP" >&2; exit 1; }
grep -q 'CLR_GFX8_TC_INVALIDATE' "$FLAGS" && grep -q 'CLR_GFX8_TC_RECORD_FENCE' "$FLAGS" || {
    echo "FATAL: CLR_GFX8_TC_* flags missing from $FLAGS" >&2; exit 1; }
echo "gfx8 TC-fence patch applied and verified (dispatch + pre-copy fence + event record)"
