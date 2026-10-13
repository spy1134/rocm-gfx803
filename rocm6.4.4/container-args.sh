# Sourced by start-comfyui.sh, gfx803, start-ollama.sh and start-chatterbox.sh: the docker run
# arguments they need, so the GPU access, user mapping and data mounts cannot
# drift apart.
# gfx803_gpu_args fills the array GFX803_ARGS with GPU access, user mapping and
# the time zone; gfx803_container_args DIR adds the ComfyUI and training mounts.
#
# RUN_AS_ROOT=1 runs the container as root instead of the invoking user.
# GFX803_COMFYUI_LISTEN=ADDRESS makes ComfyUI reachable beyond this machine
# (0.0.0.0 for every interface); by default it accepts local connections only.
# GFX803_RUNTIME=patched loads the patched HSA runtime instead of AMD's stock
# one, which the images preload by default; the patched one page-faults in some
# workloads that free and reallocate GPU memory, so it is for comparison only.
# GFX803_DEBUG=1 serializes kernel launches so a GPU fault points at the kernel
# that caused it; it slows everything down and can hide timing-dependent bugs,
# so it is for attributing a fault, not for normal use.

gfx803_gpu_args() {
  local gid
  GFX803_ARGS=(--device=/dev/kfd --device=/dev/dri --group-add video)

  # A non-root user also needs the groups that own the GPU device nodes, which
  # differ between distributions, so they are read from the host's own devices.
  if [ "${RUN_AS_ROOT:-0}" != "1" ]; then
    GFX803_ARGS+=(--user "$(id -u):$(id -g)" -e HOME=/home/gfx803)
    for gid in $(stat -c %g /dev/kfd /dev/dri/renderD* 2>/dev/null | sort -u); do
      GFX803_ARGS+=(--group-add "$gid")
    done
  fi
  # The image's clock is UTC. The host's zone file is passed in by path rather
  # than by name: Ubuntu moved the older names (US/Pacific and the like) out of
  # its tzdata package, so a name valid on the host can be missing in the image.
  if [ -e /etc/localtime ]; then
    GFX803_ARGS+=(-v "$(readlink -f /etc/localtime):/etc/host-localtime:ro" -e TZ=:/etc/host-localtime)
  fi
  case "${GFX803_RUNTIME:-stock}" in
    stock) ;;
    patched) GFX803_ARGS+=(-e LD_PRELOAD=/opt/rocm/lib/libgfx803_sgemm_shim_batched.so) ;;
    *) echo "GFX803_RUNTIME must be stock or patched" >&2; return 1 ;;
  esac
  if [ "${GFX803_DEBUG:-0}" = "1" ]; then
    GFX803_ARGS+=(-e AMD_SERIALIZE_KERNEL=3 -e HIP_LAUNCH_BLOCKING=1)
  fi
}

gfx803_container_args() {
  local dir=$1
  gfx803_gpu_args || return 1

  # Created here first: a bind-mount source Docker has to create itself ends
  # up owned by root.
  mkdir -p "$dir/workspace" \
           "$dir/ComfyUI/models" "$dir/ComfyUI/custom_nodes" \
           "$dir/ComfyUI/output" "$dir/ComfyUI/input" "$dir/ComfyUI/user" \
           "$dir/miopen/config" "$dir/miopen/cache"

  # The MIOpen folders keep its find database and compiled-kernel cache across
  # containers; without them every new container searches and compiles again.
  # They are mounted at fixed paths so they don't depend on the user's home.
  GFX803_ARGS+=(
    -e MIOPEN_USER_DB_PATH=/miopen/config -e MIOPEN_CUSTOM_CACHE_DIR=/miopen/cache
    -v "$dir/miopen/config:/miopen/config"
    -v "$dir/miopen/cache:/miopen/cache"
    -v "$dir/workspace:/workspace/"
    -v "$dir/ComfyUI/models:/opt/ComfyUI/models"
    -v "$dir/ComfyUI/custom_nodes:/opt/ComfyUI/custom_nodes"
    -v "$dir/ComfyUI/output:/opt/ComfyUI/output"
    -v "$dir/ComfyUI/input:/opt/ComfyUI/input"
    -v "$dir/ComfyUI/user:/opt/ComfyUI/user"
  )
  return 0
}

# gfx803_service_status NAME [URL]: one line for a container (its docker status),
# plus whether URL answers. Exits non-zero when the container is not running, so
# a script can use it as a check.
gfx803_service_status() {
  local name=$1 url=${2:-} st
  st=$(docker ps -a --filter "name=^${name}\$" --format '{{.Status}}')
  if [ -z "$st" ]; then echo "$name: not created"; return 1; fi
  case $st in Up*) ;; *) echo "$name: stopped ($st)"; return 1 ;; esac
  if [ -z "$url" ]; then echo "$name: running ($st)"
  elif curl -fsS -m 3 -o /dev/null "$url" 2>/dev/null; then echo "$name: running ($st), answering at $url"
  else echo "$name: running ($st), not answering yet at $url"; fi
}
