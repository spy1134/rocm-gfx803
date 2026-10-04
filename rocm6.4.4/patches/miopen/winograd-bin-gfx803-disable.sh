#!/bin/sh
set -eu
SRC="${1:-/miopen-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/winograd-bin-gfx803-disable.patch"
[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi
git -C "$SRC" apply --verbose "$PATCH"
for f in src/solver/conv/conv_bin_winoRxS.cpp src/solver/conv/conv_bin_wino3x3U.cpp; do
    count=$(grep -c 'WINOGRAD_BIN_GFX803_DISABLE_PATCH' "$SRC/$f" || true)
    if [ "$count" -ne 1 ]; then
        echo "FATAL: marker not found in $f after git apply reported success" >&2
        exit 1
    fi
done
echo "Winograd binary-solver gfx803 disable patch applied and verified"
