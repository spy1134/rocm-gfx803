set -eu

# SafeSmallKTopK (topk-radix-tiebreak-nondeterministic.patch) takes dynamic shared
# memory that grows with K. For unsorted output K is not capped, and above about
# 7800 the request exceeds the 64 KiB LDS limit. This patch sends those requests to
# RadixTopK. It must run after topk-radix-tiebreak-nondeterministic.sh.

SRC="${1:-/onnxruntime}"
SELF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PATCH="$SELF_DIR/topk-safe-small-k-smem-bound.patch"
FILE="$SRC/onnxruntime/core/providers/cuda/math/topk_impl.cuh"

[ -f "$PATCH" ] || { echo "FATAL: no patch file at $PATCH" >&2; exit 1; }
[ -d "$SRC/.git" ] || { echo "FATAL: $SRC is not a git checkout" >&2; exit 1; }
[ -f "$FILE" ] || { echo "FATAL: $FILE does not exist" >&2; exit 1; }
grep -q 'SAFE_SMALL_K_TOPK_PATCH' "$FILE" || {
    echo "FATAL: topk-radix-tiebreak-nondeterministic.patch is not applied; this one builds on it" >&2
    exit 1
}

if git -C "$SRC" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "already patched, skipping"
    exit 0
fi

git -C "$SRC" apply --verbose "$PATCH"

count=$(grep -c 'SAFE_SMALL_K_SMEM_BOUND' "$FILE" || true)
if [ "$count" -ne 1 ]; then
    echo "FATAL: expected 1 marker occurrence, found $count after git apply reported success" >&2
    exit 1
fi
echo "TopK shared-memory bound patch applied and verified in $FILE"
