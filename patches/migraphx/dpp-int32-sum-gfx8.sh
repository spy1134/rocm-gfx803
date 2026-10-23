#!/bin/sh
# Apply dpp-int32-sum-gfx8.patch (see that file for the full WHY/WHAT) via `git apply`, then
# verify the marker is present. The marker check also runs when the tree already
# carries the patch (for example after a generic loop over patches/migraphx/*.patch),
# so a tree without the change can never pass.
set -eu

SRC="${1:-/migraphx-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/dpp-int32-sum-gfx8.patch"
FILE="$SRC/src/targets/gpu/kernels/include/migraphx/kernels/reduce.hpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping apply"
else
    git -C "$SRC" apply --verbose "$PATCH"
fi

count=$(grep -c 'GFX803_DPP_INT32_SUM' "$FILE" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: expected 1 occurrence of GFX803_DPP_INT32_SUM in $FILE, found $count" >&2
    exit 1
fi
echo "MIGraphX gfx8 integer DPP sum patch applied and verified"
