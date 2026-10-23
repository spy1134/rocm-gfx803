#!/bin/sh
# Apply bn-spatial-variant1-unaligned-hw.patch (see that file for the full
# WHY/WHAT) via `git apply`, then verify the hunks actually landed.
set -eu

SRC="${1:-/miopen-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/bn-spatial-variant1-unaligned-hw.patch"
BWD_FILE="$SRC/src/kernels/MIOpenBatchNormBwdSpatial.cl"
FWD_FILE="$SRC/src/kernels/MIOpenBatchNormFwdTrainSpatial.cl"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

for f in "$BWD_FILE" "$FWD_FILE"; do
    count=$(grep -c 'GFX803_BN_VARIANT1_UNALIGNED_HW' "$f" || true)
    if [ "$count" -ne 1 ]; then
        echo "FATAL: marker not found in $f after git apply reported success" >&2
        exit 1
    fi
done
echo "BatchNorm spatial variant-1 unaligned-HW patch applied and verified"
