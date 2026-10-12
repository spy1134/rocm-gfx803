#!/bin/bash
# Checks this machine for what rocm-gfx803 needs, then builds the Docker images.
#
#   ./setup.sh                  check, ask what to set up, build it
#   ./setup.sh --check          only check the machine
#   ./setup.sh --all            build everything without asking which parts
#   ./setup.sh --comfyui        image making and LoRA training only
#   ./setup.sh --ollama         local language models only
#   ./setup.sh --yes ...        also skip the "this takes hours" confirmation
#
# Safe to run again at any time: Docker keeps every finished build step, so a
# second run, or a run after an interruption, only does what is left.
# Nothing here uses sudo; when something needs it, the script says which
# command to run.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$DIR/setup-logs"
BASE=rocm-gfx803:rocm6.4.4
COMFY=rocm-gfx803-comfyui:latest
OLLAMA=rocm-gfx803-ollama:latest

if [ -t 1 ]; then B=$'\e[1m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; N=$'\e[0m'; else B=; G=; Y=; R=; N=; fi
ok()   { echo "  ${G}ok${N}    $*"; }
warn() { echo "  ${Y}note${N}  $*"; WARNINGS=$((WARNINGS + 1)); }
fail() { echo "  ${R}fix${N}   $*"; PROBLEMS=$((PROBLEMS + 1)); }
hint() { echo "        $*"; }
ask()  { local a; read -r -p "$1 " a < /dev/tty || a=; echo "$a"; }

MODE=ask; YES=0
for arg in "$@"; do
  case "$arg" in
    --check) MODE=check ;;
    --all) MODE=all ;;
    --comfyui) MODE=comfyui ;;
    --ollama) MODE=ollama ;;
    --yes|-y) YES=1 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "setup.sh: unknown option $arg (see --help)" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- checks
PROBLEMS=0; WARNINGS=0
echo "${B}Checking this machine${N}"

if ! command -v docker >/dev/null; then
  fail "Docker is not installed."
  hint "Arch:          sudo pacman -S docker docker-buildx"
  hint "Ubuntu/Debian: sudo apt install docker.io docker-buildx"
  hint "Fedora:        sudo dnf install moby-engine docker-buildx"
  hint "then:          sudo systemctl enable --now docker"
elif ! docker info >/dev/null 2>&1; then
  if ! docker info 2>&1 | grep -qi "permission denied"; then
    fail "Docker is installed but not running."
    hint "Start it:  sudo systemctl enable --now docker"
  else
    fail "Your user is not allowed to use Docker yet."
    hint "Run:  sudo usermod -aG docker $USER"
    hint "then log out and back in (or reboot) so it takes effect."
  fi
else
  ok "Docker $(docker info --format '{{.ServerVersion}}' 2>/dev/null) is installed and usable"
  if docker buildx version >/dev/null 2>&1; then
    ok "Docker buildx is available"
  else
    fail "Docker's build plugin (buildx) is missing; the builds need it."
    hint "Arch: sudo pacman -S docker-buildx   Ubuntu/Debian: sudo apt install docker-buildx"
  fi
fi

if [ -e /dev/kfd ] && ls /dev/dri/renderD* >/dev/null 2>&1; then
  ok "GPU compute device nodes are present (/dev/kfd, /dev/dri)"
else
  fail "The GPU compute devices (/dev/kfd, /dev/dri/renderD*) are missing."
  hint "The amdgpu kernel driver must be loaded. Most distributions load it"
  hint "automatically; check 'lsmod | grep amdgpu' and that the card is seated."
fi

# Polaris 10/11 (RX 470-590, 460/560), Polaris 12 (RX 540/550), Vega M and
# Fiji (R9 Fury/Nano) are all gfx803.
card_name=""; found_amd=0
for dev in /sys/class/drm/card[0-9]*/device; do
  [ "$(cat "$dev/vendor" 2>/dev/null)" = 0x1002 ] || continue
  found_amd=1
  id=$(( $(cat "$dev/device") ))
  if { [ $id -ge $((0x67C0)) ] && [ $id -le $((0x67FF)) ]; } || { [ $id -ge $((0x6980)) ] && [ $id -le $((0x699F)) ]; } \
     || { [ $id -ge $((0x694C)) ] && [ $id -le $((0x694F)) ]; } || [ $id -eq $((0x7300)) ] || [ $id -eq $((0x730F)) ]; then
    card_name=$(printf '0x%04x' $id)
    vram=$(( $(cat "$dev/mem_info_vram_total" 2>/dev/null || echo 0) / 1048576 ))
    break
  fi
done
if [ -n "$card_name" ]; then
  ok "Found a gfx803 (Polaris-family) AMD card (PCI ID $card_name, $((vram / 1024)) GB VRAM)"
  [ "${vram:-0}" -gt 0 ] && [ "$vram" -lt 7000 ] && warn "Less than 8 GB of VRAM: image generation works, but LoRA training will not fit."
elif [ "$found_amd" = 1 ]; then
  warn "Found an AMD card, but not one of the gfx803 (Polaris-family) models this project targets."
  hint "It may not work; everything here is built and tested for gfx803 only."
else
  fail "No AMD graphics card found."
fi

kver=$(uname -r)
case "$kver" in
  6.12.*|6.13.*) warn "Linux $kver: kernels 6.12 and 6.13 are reported to crash with these cards; an LTS kernel is safer." ;;
  *) ok "Linux kernel $kver" ;;
esac

mem_gb=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)
if [ "$mem_gb" -lt 15 ]; then
  warn "${mem_gb} GB of RAM: the build will run, but slowly (it compiles fewer files at once)."
else
  ok "${mem_gb} GB of RAM"
fi

root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
free_gb=$(df -P -BG "$root" 2>/dev/null | awk 'NR == 2 {gsub("G", "", $4); print $4}')
free_gb=${free_gb:-$(df -P -BG / | awk 'NR == 2 {gsub("G", "", $4); print $4}')}
if [ "${free_gb:-0}" -lt 40 ]; then
  fail "Only ${free_gb} GB free where Docker stores images ($root); the build needs about 100 GB."
elif [ "$free_gb" -lt 100 ]; then
  warn "${free_gb} GB free where Docker stores images; about 100 GB is recommended for a first build."
else
  ok "${free_gb} GB free disk space for Docker"
fi

have() { docker image inspect "$1" >/dev/null 2>&1; }
echo
echo "${B}Already built${N}"
images=("$BASE" "$COMFY" "$OLLAMA")
labels=("the base" "image making and training" "language models")
for i in 0 1 2; do
  if have "${images[$i]}"; then ok "${labels[$i]} (${images[$i]})"; else echo "  -     ${labels[$i]} (not yet)"; fi
done

echo
if [ "$PROBLEMS" -gt 0 ]; then
  echo "${R}${PROBLEMS} thing(s) to fix first${N}, listed above. Run ./setup.sh again afterwards."
  exit 1
fi
echo "${G}Ready.${N}$([ "$WARNINGS" -gt 0 ] && echo " ($WARNINGS note(s) above are worth reading.)")"
[ "$MODE" = check ] && exit 0

# ---------------------------------------------------------------- choose
if [ ! -t 0 ] && { [ "$MODE" = ask ] || [ "$YES" = 0 ]; }; then
  echo "No terminal to ask questions on: pass what to build and --yes, e.g. ./setup.sh --all --yes"
  exit 2
fi
if [ "$MODE" = ask ]; then
  echo
  echo "${B}What would you like to set up?${N}"
  echo "  1) Everything (recommended)"
  echo "  2) Image making and LoRA training (ComfyUI, sd-scripts)"
  echo "  3) Local language models (Ollama)"
  case "$(ask 'Choose 1, 2 or 3 [1]:')" in
    2) MODE=comfyui ;; 3) MODE=ollama ;; ""|1) MODE=all ;;
    *) echo "Nothing chosen; run ./setup.sh again when ready."; exit 0 ;;
  esac
fi
TARGETS=(base)
case "$MODE" in all) TARGETS+=(comfyui ollama) ;; comfyui) TARGETS+=(comfyui) ;; ollama) TARGETS+=(ollama) ;; esac

if ! have "$BASE" && [ "$YES" = 0 ]; then
  echo
  echo "The first build compiles the GPU libraries for your card from source. It takes"
  echo "several hours and keeps the CPU busy; the computer stays usable meanwhile."
  echo "You can stop it with Ctrl+C at any point and run ./setup.sh again to continue."
  case "$(ask 'Start now? [Y/n]:')" in n|N|no|No) echo "Okay, nothing built."; exit 0 ;; esac
fi

# ---------------------------------------------------------------- build
mkdir -p "$LOGS"
build() {
  local tag=$1 file=$2 label=$3 log="$LOGS/$4.log" start=$SECONDS
  echo
  echo "${B}Building $label${N}  (log: ${log#$DIR/})"
  # Plain progress, filtered to one line per build step, keeps the output
  # readable; the full log has everything if something goes wrong. Success is
  # docker build's own exit status: the filter finds nothing to show when every
  # step comes from the cache.
  docker build --progress=plain -t "$tag" -f "$DIR/$file" "$DIR" 2>&1 | tee "$log" \
    | { grep --line-buffered -E '^#[0-9]+ \[[^]]+\] ' | sed -u -E 's/^#[0-9]+ /  /' | cut -c1-100; true; }
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    echo "  ${G}done${N} in $(( (SECONDS - start) / 60 )) min"
    return 0
  fi
  echo "  ${R}failed${N}. The last lines of the log:"
  tail -n 25 "$log" | sed 's/^/    /'
  echo "  Full log: $log"
  return 1
}
for t in "${TARGETS[@]}"; do
  case "$t" in
    base)    build "$BASE"   Dockerfile         "the base (GPU libraries, PyTorch)" base || exit 1 ;;
    comfyui) build "$COMFY"  Dockerfile-ComfyUI "image making and training"         comfyui || exit 1 ;;
    ollama)  build "$OLLAMA" Dockerfile-Ollama  "language models (Ollama)"          ollama || exit 1 ;;
  esac
done

# ---------------------------------------------------------------- first GPU check
echo
echo "${B}Checking that the containers can use your GPU${N}"
. "$DIR/container-args.sh"
gfx803_gpu_args
if gpu=$(docker run --rm "${GFX803_ARGS[@]}" "$BASE" python3 -c \
    "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>/dev/null | tail -n 1) \
   && [ -n "$gpu" ]; then
  ok "PyTorch sees: $gpu"
else
  echo "  ${R}The GPU is not visible inside the container.${N} Check the notes above, and that"
  echo "  no other program holds the GPU exclusively; then run ./setup.sh --check."
  exit 1
fi

# ---------------------------------------------------------------- next steps
echo
echo "${G}${B}All set.${N} What to run next, from this folder:"
for t in "${TARGETS[@]}"; do
  case "$t" in
    comfyui)
      echo "  ./start-comfyui.sh        start ComfyUI (then open http://localhost:8188 in a browser)"
      echo "  ./start-comfyui.sh bash   a shell in the container, for gfx803-train (LoRA training)" ;;
    ollama)
      echo "  ./start-ollama.sh         start the language-model server (http://127.0.0.1:11434)"
      echo "  ./start-ollama.sh pull qwen3:8b    download a model (Qwen3 8B fits well in 8 GB)"
      echo "  ./start-ollama.sh run qwen3:8b     chat with it in the terminal" ;;
  esac
done
echo "  README.md in this folder explains everything in more detail."
