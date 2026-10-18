#!/bin/bash
# install.sh
# Installs or links the gfx803 ComfyUI custom node into the target custom_nodes folder.
# Usage: ./install.sh [target_custom_nodes_path]
# Default target: rocm6.4.4/ComfyUI/custom_nodes (relative to repository root)

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "${SELF_DIR}/.." && pwd)"
DEFAULT_TARGET="${TOOLS_DIR}/../rocm6.4.4/ComfyUI/custom_nodes"
TARGET="${1:-$DEFAULT_TARGET}"
NODE_DEST="${TARGET}/ComfyUI-gfx803-attention"

echo "[install.sh] Target directory: ${NODE_DEST}"
mkdir -p "${NODE_DEST}"

# The host usually has no ROCm of its own; the base image's hipcc is the
# compiler the ComfyUI image runs against anyway. GFX803_IMAGE overrides it.
IMAGE="${GFX803_IMAGE:-rocm-gfx803:rocm6.4.4}"
build() {
    local out="$1" src="$2"
    [ -f "${TOOLS_DIR}/${src}" ] || return 0
    [ "${TOOLS_DIR}/${out}" -nt "${TOOLS_DIR}/${src}" ] && return 0
    echo "[install.sh] Compiling ${out}..."
    if command -v hipcc >/dev/null; then
        hipcc -O3 -shared -fPIC --offload-arch=gfx803 -o "${TOOLS_DIR}/${out}" "${TOOLS_DIR}/${src}"
    else
        docker run --rm -u "$(id -u):$(id -g)" -v "${TOOLS_DIR}:/src" -w /src --entrypoint hipcc "${IMAGE}" \
            -O3 -shared -fPIC --offload-arch=gfx803 -o "${out}" "${src}"
    fi
}
build libgfx803_attn.so gfx803_fused_attention.hip
build libgfx803_geglu.so gfx803_geglu.hip
build libgfx803_norm.so gfx803_group_norm_silu.hip

# Copy python loaders and init
cp "${SELF_DIR}/__init__.py" "${NODE_DEST}/"
cp "${TOOLS_DIR}/gfx803_attention.py" "${NODE_DEST}/"
cp "${TOOLS_DIR}/gfx803_geglu.py" "${NODE_DEST}/"
cp "${TOOLS_DIR}/gfx803_norm.py" "${NODE_DEST}/"

# Copy compiled shared libraries if present in tools/
for lib in libgfx803_attn.so libgfx803_geglu.so libgfx803_norm.so; do
    if [ -f "${TOOLS_DIR}/${lib}" ]; then
        cp "${TOOLS_DIR}/${lib}" "${NODE_DEST}/"
        echo "[install.sh] Copied ${lib}"
    else
        echo "[install.sh] ERROR: ${TOOLS_DIR}/${lib} could not be built or found." >&2
        exit 1
    fi
done

echo "[install.sh] Successfully installed ComfyUI-gfx803-attention to ${NODE_DEST}"
