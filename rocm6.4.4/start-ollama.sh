#!/bin/bash
# Runs the rocm-gfx803 Ollama image as a background service that only this
# machine can reach, at http://127.0.0.1:11434 (the Ollama and OpenAI-compatible
# APIs, e.g. for SillyTavern or a coding agent running on the host).
#
#   ./start-ollama.sh               start the service (does nothing if running)
#   ./start-ollama.sh --stop        stop it
#   ./start-ollama.sh --logs        follow its log
#   ./start-ollama.sh --status      show whether it runs and which model is loaded
#   ./start-ollama.sh --list        list the installed models (does not start the service)
#   ./start-ollama.sh pull qwen3:8b any other arguments run the ollama CLI in it
#
# OLLAMA_DATA   models and Ollama's key (default: ollama/ next to this script)
# OLLAMA_PORT   host port (default 11434)
# OLLAMA_KV_CACHE_TYPE  default q8_0; f16 for full precision
# OLLAMA_KEEP_ALIVE  default -1 (a loaded model stays loaded until it is
#               unloaded with `ollama stop MODEL`); Ollama's
#               own 5-minute default drops it between agent sessions, and a
#               reload costs tens of seconds to minutes for the large models
# OLLAMA_FLASH_ATTENTION, OLLAMA_CONTEXT_LENGTH, OLLAMA_DEBUG
#               passed to the server when set
# GFX803_OLLAMA_HEADROOM_MIB  VRAM kept free for the desktop beyond what it
#               uses when the server starts (default 512); the card's in-use
#               VRAM is read at start, so start the server with your usual
#               programs open. LLAMA_ARG_FIT_TARGET (MiB) overrides the total.
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
  --status) . "$SCRIPT_DIR/container-args.sh"
            gfx803_service_status "$NAME" "http://127.0.0.1:$PORT/api/version" || exit 1
            docker exec "$NAME" ollama ps 2>/dev/null; exit 0 ;;
  --list) running || die "the service is not running; start it first"
          exec docker exec "$NAME" ollama list ;;
  -h|--help) sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
  export OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:--1}
  # llama.cpp fits each model to the VRAM that ROCm reports free and keeps a
  # margin (1 GiB by default). On this card ROCm's figure leaves out the
  # display's framebuffers and other graphics programs' buffers, so with a
  # desktop running most of that margin is already taken, and a model plus its
  # working buffers can fill the card; the compositor then cannot pin its
  # framebuffers and the desktop stalls. amdgpu's own counter does include
  # them, so the margin becomes what is in use now plus room for the desktop
  # to grow. Memory held by other ROCm processes (a TTS server, ComfyUI) is
  # already missing from ROCm's free figure; counting it in the margin too
  # would keep that much VRAM idle, so it is subtracted (KFD's per-process
  # counters; this assumes one AMD GPU). With no desktop (display manager
  # stopped) the margin is close to the headroom alone, which leaves the most
  # room for context.
  if [ -z "${LLAMA_ARG_FIT_TARGET:-}" ]; then
    for used in /sys/class/drm/card*/device/mem_info_vram_used; do
      [ "$(cat "${used%/*}/vendor" 2>/dev/null)" = 0x1002 ] || continue
      rocm_bytes=0
      for f in /sys/class/kfd/kfd/proc/*/vram_*; do
        [ -r "$f" ] && rocm_bytes=$(( rocm_bytes + $(cat "$f") ))
      done
      export LLAMA_ARG_FIT_TARGET=$(( ($(cat "$used") - rocm_bytes) / 1048576 + ${GFX803_OLLAMA_HEADROOM_MIB:-512} ))
      break
    done
  fi
  for v in OLLAMA_KV_CACHE_TYPE OLLAMA_FLASH_ATTENTION OLLAMA_CONTEXT_LENGTH OLLAMA_KEEP_ALIVE OLLAMA_DEBUG LLAMA_ARG_FIT_TARGET; do
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
