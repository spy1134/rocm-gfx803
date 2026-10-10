#!/bin/bash
# Runs the rocm-gfx803 Ollama image as a background service that only this
# machine can reach, at http://127.0.0.1:11434 (the Ollama and OpenAI-compatible
# APIs, e.g. for SillyTavern or a coding agent running on the host).
#
#   ./start-ollama.sh               start the service (does nothing if running)
#   ./start-ollama.sh --stop        stop it
#   ./start-ollama.sh --logs        follow its log
#   ./start-ollama.sh pull qwen3:8b any other arguments run the ollama CLI in it
#
# OLLAMA_DATA   models and Ollama's key (default: ollama/ next to this script)
# OLLAMA_PORT   host port (default 11434)
# GFX803_RUNTIME=patched  load the base image's patched HSA runtime instead of
#               AMD's stock one (the default); its va-reuse-defer change faults
#               ("Page not present") in some workloads that free and reallocate
#               GPU memory, llama-server among them, so it is for comparison only
# OLLAMA_KV_CACHE_TYPE  default q8_0; f16 for full precision
# OLLAMA_FLASH_ATTENTION, OLLAMA_CONTEXT_LENGTH, OLLAMA_KEEP_ALIVE
#               passed to the server when set
#
# The 8-bit KV cache halves the context's VRAM, which on an 8 GB card decides
# whether a model's context fits beside it at all: Qwen3 8B at 16k needs 2.3 GB
# of f16 cache and spills layers to the CPU, but runs fully on the GPU with
# q8_0. Measured on an RX 590 (llama-bench, Qwen3 8B): prompt speed unchanged,
# generation 3% slower on an empty context and faster from 4k tokens deep
# (26.0 vs 24.9 tok/s at 4k, 24.4 vs 24.0 at 8k), where the gfx803 q8_0 decode
# kernel (patches/ollama/llama-cpp/102-*) beats the f16 path.
#
# The port is published on the host's loopback address only. The server inside
# the container listens on all of the container's interfaces, because Docker
# forwards published ports to the container's own network address, not to its
# loopback.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=${OLLAMA_IMAGE:-rocm-gfx803-ollama:latest}
NAME=ollama
DATA=${OLLAMA_DATA:-$SCRIPT_DIR/ollama}
PORT=${OLLAMA_PORT:-11434}

die() { echo "start-ollama: $*" >&2; exit 1; }
running() { [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ]; }

case "${1:-}" in
  --stop) running && docker stop "$NAME" >/dev/null; exit 0 ;;
  --logs) exec docker logs -f "$NAME" ;;
  -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

if ! running; then
  docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found; build it first (see Dockerfile-Ollama)"
  # shellcheck source=container-args.sh
  . "$SCRIPT_DIR/container-args.sh"
  gfx803_gpu_args
  # Created here first: a bind-mount source Docker has to create itself ends
  # up owned by root.
  mkdir -p "$DATA"
  export OLLAMA_KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE:-q8_0}
  for v in OLLAMA_KV_CACHE_TYPE OLLAMA_FLASH_ATTENTION OLLAMA_CONTEXT_LENGTH OLLAMA_KEEP_ALIVE; do
    [ -n "${!v:-}" ] && GFX803_ARGS+=(-e "$v")
  done
  home=/home/gfx803
  [ "${RUN_AS_ROOT:-0}" = "1" ] && home=/root
  docker rm "$NAME" >/dev/null 2>&1 || true
  docker run -d --init --name "$NAME" \
    "${GFX803_ARGS[@]}" \
    -v "$DATA:$home/.ollama" \
    -p "127.0.0.1:$PORT:11434" \
    "$IMAGE" >/dev/null
  STARTED=1
  echo "Ollama is starting on http://127.0.0.1:$PORT (models in $DATA)"
  if docker ps --format '{{.Names}}' | grep -qx gfx803; then
    echo "start-ollama: the gfx803 service is running too; a loaded model and a training job share the card's 8 GB" >&2
  fi
fi

if [ $# -gt 0 ]; then
  # A freshly started server needs about 25 s (on an RX 590 host) before it
  # answers, mostly GPU discovery; until then a CLI command either hangs
  # silently or fails to connect, depending on how far startup got.
  if [ "${STARTED:-0}" = 1 ]; then
    echo "Waiting for the Ollama server to be ready (about half a minute)..."
    for _ in $(seq 90); do
      docker exec "$NAME" ollama list >/dev/null 2>&1 && break
      sleep 1
    done
  fi
  exec docker exec -it "$NAME" ollama "$@"
fi
