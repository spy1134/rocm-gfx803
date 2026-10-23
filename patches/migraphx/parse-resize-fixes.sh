#!/bin/sh
# Apply parse-resize-fixes.patch (see that file for the full WHY/WHAT) via `git apply`, then
# verify both fixes are present. The checks also run when the tree already carries the
# patch (for example after a generic loop over patches/migraphx/*.patch), so a tree
# without the change can never pass.
set -eu

SRC="${1:-/migraphx-src}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/parse-resize-fixes.patch"
FILE="$SRC/src/onnx/parse_resize.cpp"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping apply"
else
    git -C "$SRC" apply --verbose "$PATCH"
fi

grep -qF 'info.attributes.at("keep_aspect_ratio_policy").s() != "stretch"' "$FILE" || {
    echo "FATAL: the keep_aspect_ratio_policy=\"stretch\" fix is not in $FILE" >&2; exit 1; }
grep -qF 'calc_neighbor_points(vvv_ind, in_s.as_standard(), out_s, resized_m)' "$FILE" || {
    echo "FATAL: the standard-strides gather-index fix is not in $FILE" >&2; exit 1; }
echo "MIGraphX Resize parser fixes applied and verified"
