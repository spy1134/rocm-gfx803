#!/bin/bash
# Runs the rocm-gfx803 ComfyUI image with the GPU passed through and its data
# kept on the host, next to this script (all of it git-ignored).
#
#   ./start-comfyui.sh              # start ComfyUI on http://localhost:8188
#   GFX803_COMFYUI_LISTEN=0.0.0.0 ./start-comfyui.sh   # ...reachable from the LAN too
#   ./start-comfyui.sh --status     show whether it runs
#   ./start-comfyui.sh bash         # a shell instead, e.g. for gfx803-train
#   GFX803_DEBUG=1 ./start-comfyui.sh ...
#   RUN_AS_ROOT=1 ./start-comfyui.sh ...
#
# The container runs as the user who starts it, so everything it writes into the
# folders below is owned by that user. See container-args.sh for the settings
# it shares with gfx803, the all-in-one launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=container-args.sh
. "$SCRIPT_DIR/container-args.sh"
[ "${1:-}" = --status ] && { gfx803_service_status comfyui "http://127.0.0.1:8188/"; exit; }
gfx803_container_args "$SCRIPT_DIR"

# The port is published on this machine's loopback address unless
# GFX803_COMFYUI_LISTEN names another: ComfyUI has no authentication, and
# ComfyUI-Manager installs and runs code.
# --init: the container's first process is a shell, and the kernel drops
# signals to PID 1 that it has no handler for. Without an init process to
# forward them, closing the terminal (or a screen session) running this
# leaves the container and whatever it was doing running.
docker run --rm -it --init \
  "${GFX803_ARGS[@]}" \
  -p "${GFX803_COMFYUI_LISTEN:-127.0.0.1}:8188:8188" \
  --name comfyui \
  rocm-gfx803-comfyui:latest "$@"
