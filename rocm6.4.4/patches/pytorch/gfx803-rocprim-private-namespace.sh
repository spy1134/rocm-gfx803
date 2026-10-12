#!/bin/sh
set -eu
SRC="${1:-/pytorch}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/gfx803-rocprim-private-namespace.patch"
TARGET="$SRC/cmake/Dependencies.cmake"
[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi
git -C "$SRC" apply --verbose "$PATCH"
count=$(grep -c 'GFX803_ROCPRIM_PRIVATE_NAMESPACE_PATCH' "$TARGET" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: marker not found in $TARGET after git apply reported success" >&2
    exit 1
fi
uses=$(grep -c 'list(APPEND HIP_CXX_FLAGS -Drocprim=at_hip_rocprim)' "$TARGET" || true)
if [ "$uses" -ne 1 ]; then
    echo "FATAL: expected one -Drocprim=at_hip_rocprim flag, found $uses" >&2
    exit 1
fi
echo "PyTorch rocPRIM private-namespace patch applied and verified"
