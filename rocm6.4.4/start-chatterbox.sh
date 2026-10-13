#!/bin/bash
# Runs the Chatterbox TTS server (gfx803-chatterbox image) at http://127.0.0.1:8004, an
# OpenAI-compatible speech endpoint: POST /v1/audio/speech. Only this machine can reach it.
#
#   ./start-chatterbox.sh          start it (does nothing if running)
#   ./start-chatterbox.sh --stop   stop it
#   ./start-chatterbox.sh --logs   follow its log
#   ./start-chatterbox.sh --status show whether it runs
#
# CHATTERBOX_T3_DTYPE  float16 (default) or float32 for the text-to-speech-token model;
#                      float16 saves about 1 GB of VRAM
# CHATTERBOX_PORT      default 8004
# CHATTERBOX_DIR       the Chatterbox project (server.py, voices/, MIOpen databases), mounted at
#                      /data (default: Chatterbox/ two levels above this script)
#
# Voices: put a 10-30 s clip of one clean speaker in voices/<name>.wav (or .mp3/.flac) and
# ask for voice "<name>"; "default" (and OpenAI names such as "alloy") is the built-in voice.
#
# Start it before Ollama: Ollama sizes its models by the VRAM free when it starts, and
# would otherwise plan to use the memory this server takes.
#
# Every request has a different length, and MIOpen tunes each new convolution shape it sees:
# its default exhaustive search added 20 s to a 6 s sentence, the fast mode 4 s. The fast
# mode is set here, and MIOpen's databases persist in miopen/ so a length is tuned only once.
script_dir=$(cd "$(dirname "$0")" && pwd)
here=$(cd "${CHATTERBOX_DIR:-$script_dir/../../Chatterbox}" && pwd) || exit 1
. "$script_dir/container-args.sh"
name=chatterbox
case "${1:-}" in
  --stop) docker stop "$name" >/dev/null 2>&1; docker rm "$name" >/dev/null 2>&1; echo "stopped"; exit 0 ;;
  --logs) exec docker logs -f "$name" ;;
  --status) gfx803_service_status "$name" "http://127.0.0.1:${CHATTERBOX_PORT:-8004}/v1/models"; exit ;;
esac
if docker ps --format '{{.Names}}' | grep -qx "$name"; then echo "already running"; exit 0; fi
docker rm "$name" >/dev/null 2>&1
mkdir -p "$here/miopen/db" "$here/miopen/cache"
gfx803_gpu_args
docker run -d --name "$name" --network host "${GFX803_ARGS[@]}" \
  -e HOME=/tmp -e PORT="${CHATTERBOX_PORT:-8004}" \
  -e CHATTERBOX_T3_DTYPE="${CHATTERBOX_T3_DTYPE:-float16}" \
  -e MIOPEN_FIND_MODE=FAST -e MIOPEN_USER_DB_PATH=/data/miopen/db -e MIOPEN_CUSTOM_CACHE_DIR=/data/miopen/cache \
  -v "$here:/data" gfx803-chatterbox:latest >/dev/null
echo "Chatterbox is starting on http://127.0.0.1:${CHATTERBOX_PORT:-8004} (model load takes about a minute)"
