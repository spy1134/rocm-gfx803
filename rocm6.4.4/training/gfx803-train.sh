#!/bin/bash
# gfx803-train.sh - SD1.5 LoRA training wrapper for the rocm-gfx803 ComfyUI
# image (gfx803 / Polaris). The image installs it as `gfx803-train`; run it
# inside the container (start-comfyui.sh bash):
#
#   gfx803-train --data <set>      # a folder under /workspace/training_data
#   gfx803-train --help
#
# What it does
#   * Loads the SGEMM shim and chosen HSA runtime (--shim, --runtime), with
#     diagnostic switches (--no-batched, --debug-kernels, --shim-verbose).
#   * Resolves the base model and VAE by name from ComfyUI's model folders.
#   * Names each run automatically: <dataset>_<model>_d<dim>_<date-time>.
#   * Records every run under /workspace/runs/<name>/ (settings, status, log)
#     and saves full training state with each checkpoint.
#   * On start, detects a run that did not finish (GPU hang, reboot, Ctrl+C)
#     and offers to resume it from the last saved state.
#   * Checks the GPU environment before training (shim actually loaded, other
#     VRAM users) and discards unreadable or NaN cached latents.
#   * Watches the loss: if it turns NaN, stops, deletes checkpoints whose
#     weights contain NaN, and resumes from the last clean state
#     (--nan-retries, default 3). Checkpoints are saved every 100 steps.
#
# Output layout
#   /workspace/output/<name>/<name>-step00000200.safetensors   checkpoints
#   /workspace/output/<name>/<name>-step00000200-state/        resume state
#   /workspace/output/<name>/<name>.safetensors                final LoRA
#   /workspace/runs/<name>/{run.env,status,train.log,tb/}      bookkeeping
# Step numbers are cumulative across resumed segments.

set -uo pipefail

# ---------------------------------------------------------------- defaults
# Edit these to change what a plain run uses.
DEFAULT_MODEL=${DEFAULT_MODEL:-SD15/v1-5-pruned-emaonly}
DEFAULT_VAE=${DEFAULT_VAE:-vae-ft-mse-840000-ema-pruned}
WORKSPACE=${WORKSPACE:-/workspace}
DATA_ROOT=${DATA_ROOT:-$WORKSPACE/training_data}
DEFAULT_DATA=${DEFAULT_DATA:-}
SD_SCRIPTS=${SD_SCRIPTS:-/opt/sd-scripts}
TRAINING_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
CKPT_DIR=${CKPT_DIR:-/opt/ComfyUI/models/checkpoints}
VAE_DIR=${VAE_DIR:-/opt/ComfyUI/models/vae}
ACCELERATE=${ACCELERATE:-accelerate}
RUNS_DIR=$WORKSPACE/runs
OUT_ROOT=$WORKSPACE/output
NEW_SHIM=${NEW_SHIM:-/opt/rocm/lib/libgfx803_sgemm_shim_batched.so}
ORIG_SHIM=${ORIG_SHIM:-/opt/rocm/lib/libgfx803_sgemm_shim.so}
STOCK_HSA=${STOCK_HSA:-/opt/rocm/lib/libhsa-runtime64.so.1.15.60404}
VRAM_WARN_MIB=${VRAM_WARN_MIB:-1024}

# Training parameters (saved per run; a resumed run reuses its own values).
MODEL_ARG=""
VAE_ARG=""
DATA_ARG=""
STEPS=1200
SAVE_EVERY=100
LR=1e-4
DIM=8
ALPHA=8
BATCH=1
GRAD_ACCUM=4
RESOLUTION=512
MIN_BUCKET=256
MAX_BUCKET=512
WARMUP=20
SCHEDULER=constant_with_warmup
INIT_WEIGHTS=""
TRAIN_TE=0
TE_LR=5e-5
EXTRA_ARGS=()

# Diagnostics (taken from the command line every time, also when resuming).
RUNTIME=stock
SHIM=new
NO_BATCHED=0
DEBUG_KERNELS=0
SHIM_VERBOSE=0
VRAM_RESERVE=${GFX803_VRAM_RESERVE:-auto}
VRAM_CAP=${GFX803_VRAM_CAP:-}

# Behaviour
RUN_NAME=""
RESUME_MODE=ask
NAN_RETRIES=${GFX803_NAN_RETRIES:-3}
SAVE_EVERY_CLI=""
DIAG_ARGS=()
RESUME_RUN=""
ASSUME_YES=0
FRESH_LATENTS=0
DRY_RUN=0
PREFLIGHT=1
ACTION=train

# ------------------------------------------------------------------ helpers
log()  { printf '[gfx803-train] %s\n' "$*"; }
warn() { printf '[gfx803-train] WARNING: %s\n' "$*" >&2; }
die()  { printf '[gfx803-train] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: gfx803-train.sh [options] [-- extra train_network.py args]

With no options on a terminal, an interactive menu picks the dataset, model
and settings (arrow keys, Enter); it ends by printing the equivalent command.
--menu opens it explicitly; GFX803_TRAIN_NO_MENU=1 turns it off.

Data and model
  -d, --data NAME|DIR      dataset: a set name under $DATA_ROOT
                           (partial names OK) or a folder path. The set holds
                           <repeats>_<trigger>/ subfolders of images + .txt captions.
                           [default: the only set there, if there is just one]
  -m, --model NAME|PATH    base checkpoint; a name is looked up under
                           $CKPT_DIR (partial names OK)
                           [default: $DEFAULT_MODEL]
      --vae NAME|PATH|none VAE; looked up under $VAE_DIR
                           [default: $DEFAULT_VAE]
  -n, --name NAME          run name [default: <dataset>_<model>_d<dim>_<date-time>]
      --init-weights FILE  start from an existing LoRA .safetensors (e.g. a
                           checkpoint from a run made before this wrapper)

Training
  -s, --steps N            total optimisation steps [$STEPS]
      --save-every N       checkpoint + resume state every N steps [$SAVE_EVERY]
      --lr LR              learning rate [$LR]
      --train-text-encoder also train LoRA weights in the text encoder (CLIP), not
                           just the UNet. Helps a trigger word or style respond to
                           prompts; overfits quickly, so it uses its own, lower rate
      --te-lr LR           text encoder learning rate with --train-text-encoder [$TE_LR]
      --dim N / --alpha N  LoRA rank / alpha [$DIM / $ALPHA]
      --batch N            batch size [$BATCH]
      --grad-accum N       gradient accumulation steps [$GRAD_ACCUM]
      --resolution N       training resolution [$RESOLUTION]
      --buckets MIN:MAX    bucket resolution range [$MIN_BUCKET:$MAX_BUCKET]
      --warmup N           LR warmup steps [$WARMUP]
      --scheduler NAME     LR scheduler [$SCHEDULER]
      --fresh-latents      delete cached latents and re-encode them

GPU stack and diagnostics
      --runtime stock|patched
                           stock: AMD's libhsa-runtime64 ($STOCK_HSA) [default]
                           patched: the image's patched one +
                           HSA_DISABLE_FRAGMENT_ALLOCATOR=0; page-faults in
                           some workloads, for comparison only
      --shim new|original|none|PATH
                           new: the image's batched shim (correct attention) [default]
                           original: the older shim (wrong attention gradients)
      --no-batched         new shim, but batched GEMMs go to rocBLAS
      --debug-kernels      AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 (slow)
      --shim-verbose N     log the shim's first N GEMM calls
      --vram-reserve auto|MIB
                           VRAM to leave for everything else; PyTorch is capped
                           at the rest. auto: what the card's other users hold
                           at startup + 1024 MiB for the HIP runtime's context
                           and loaded kernels. Warns when a display on the card
                           is left little headroom; stops only when the cap
                           would be under 3 GiB [\$GFX803_VRAM_RESERVE or auto]
      --vram-cap MIB|none  cap PyTorch at exactly MIB, or not at all; overrides
                           --vram-reserve. Uncapped, PyTorch's cache can fill
                           VRAM and push a desktop on the card out
                           [\$GFX803_VRAM_CAP or unset]
      --no-preflight       skip the GPU/latent checks before training
      --nan-retries N      when the loss turns NaN: stop, drop the NaN
                           checkpoints and resume from the last clean state,
                           up to N times per run [3]

Runs and resuming
      --resume ask|auto|never
                           what to do when an unfinished run is found [ask]
      --resume-run NAME    resume this run explicitly
  -y, --yes                answer yes to prompts
      --list-models        list checkpoints and VAEs
      --list-data          list datasets under $DATA_ROOT
      --list-runs          list recorded runs and their progress
      --print-defaults     print the folders and default settings as KEY=VALUE
                           (used by the menu and the queue dashboard)
      --dry-run            show what would run, change nothing
  -h, --help
EOF
}

confirm() {  # confirm "question" -> 0 for yes
    if [ "$ASSUME_YES" = 1 ]; then return 0; fi
    if [ ! -t 0 ]; then return 0; fi
    local ans
    read -r -p "$1 [Y/n] " ans
    [[ -z "$ans" || "$ans" =~ ^[Yy] ]]
}

sanitize() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_' | sed 's/__*/_/g; s/^_//; s/_$//'; }

step_of() {  # extract the 8-digit step number from a checkpoint/state name
    if [[ "$1" =~ -step([0-9]{8})(-state)?(\.safetensors)?$ ]]; then
        echo $((10#${BASH_REMATCH[1]}))
    else
        echo -1
    fi
}

# ------------------------------------------------------------ model lookup
find_model() {  # find_model <dir> <name|path> <label>
    local dir=$1 query=$2 label=$3
    if [ -f "$query" ]; then readlink -f "$query"; return 0; fi
    [ -d "$dir" ] || die "$label folder not found: $dir"
    local q=${query%.safetensors}; q=${q%.ckpt}; q=${q%.pt}
    local -a all=() exact=() partial=()
    mapfile -t all < <(find "$dir" -type f \( -name '*.safetensors' -o -name '*.ckpt' -o -name '*.pt' \) | sort)
    local f rel base
    for f in "${all[@]}"; do
        rel=${f#"$dir"/}; rel=${rel%.*}; base=$(basename "$rel")
        if [ "$rel" = "$q" ] || [ "$base" = "$q" ]; then exact+=("$f")
        elif [[ "${rel,,}" == *"${q,,}"* ]]; then partial+=("$f"); fi
    done
    if [ ${#exact[@]} -eq 1 ]; then echo "${exact[0]}"; return 0; fi
    if [ ${#exact[@]} -eq 0 ] && [ ${#partial[@]} -eq 1 ]; then echo "${partial[0]}"; return 0; fi
    local -a matches=("${exact[@]}" "${partial[@]}")
    if [ ${#matches[@]} -eq 0 ]; then
        die "no $label matches '$query' under $dir (see --list-models)"
    fi
    printf '[gfx803-train] ERROR: "%s" matches several %s files:\n' "$query" "$label" >&2
    printf '    %s\n' "${matches[@]#"$dir"/}" >&2
    exit 1
}

list_data() {
    echo "Datasets under $DATA_ROOT:"
    local d sub n
    for d in "$DATA_ROOT"/*/; do
        [ -d "$d" ] || continue
        printf '    %s\n' "$(basename "$d")"
        for sub in "$d"[0-9]*_*/; do
            [ -d "$sub" ] || continue
            n=$(find "$sub" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l)
            printf '        %-30s %s images\n' "$(basename "$sub")" "$n"
        done
    done
}

find_data() {  # find_data <name|path>  (empty: the only set under DATA_ROOT)
    local q=$1 d
    local -a sets=() matches=()
    if [ -n "$q" ] && [ -d "$q" ]; then readlink -f "$q"; return 0; fi
    for d in "$DATA_ROOT"/*/; do [ -d "$d" ] && sets+=("${d%/}"); done
    if [ -z "$q" ]; then
        [ ${#sets[@]} -eq 1 ] && { echo "${sets[0]}"; return 0; }
        [ ${#sets[@]} -eq 0 ] && die "no datasets under $DATA_ROOT"
        die "several datasets under $DATA_ROOT; pick one with --data (see --list-data)"
    fi
    for d in "${sets[@]}"; do
        [ "$(basename "$d")" = "$q" ] && { echo "$d"; return 0; }
        [[ "$(basename "$d" | tr '[:upper:]' '[:lower:]')" == *"$(echo "$q" | tr '[:upper:]' '[:lower:]')"* ]] && matches+=("$d")
    done
    [ ${#matches[@]} -eq 1 ] && { echo "${matches[0]}"; return 0; }
    [ ${#matches[@]} -eq 0 ] && die "no dataset matches '$q' under $DATA_ROOT (see --list-data)"
    die "'$q' matches several datasets: $(printf '%s ' "${matches[@]##*/}")"
}

list_models() {
    echo "Checkpoints under $CKPT_DIR:"
    find "$CKPT_DIR" -type f \( -name '*.safetensors' -o -name '*.ckpt' \) 2>/dev/null \
        | sed "s|^$CKPT_DIR/|    |" | sort
    echo "VAEs under $VAE_DIR:"
    find "$VAE_DIR" -type f \( -name '*.safetensors' -o -name '*.ckpt' -o -name '*.pt' \) 2>/dev/null \
        | sed "s|^$VAE_DIR/|    |" | sort
}

# ------------------------------------------------------------ run records
run_dir() { echo "$RUNS_DIR/$1"; }
out_dir() { echo "$OUT_ROOT/$1"; }

status_get() {  # status_get <run> <KEY>
    local f; f="$(run_dir "$1")/status"
    [ -f "$f" ] && sed -n "s/^$2=//p" "$f" | tail -n 1
}

status_set() {  # status_set <run> KEY=VALUE ...
    local f; f="$(run_dir "$1")/status"; shift
    local kv key tmp
    tmp=$(mktemp)
    [ -f "$f" ] && cp "$f" "$tmp"
    for kv in "$@" "UPDATED=$(date +%s)"; do
        key=${kv%%=*}
        grep -v "^$key=" "$tmp" > "$tmp.new" || true
        echo "$kv" >> "$tmp.new"
        mv "$tmp.new" "$tmp"
    done
    mv "$tmp" "$f"
    sync "$f" 2>/dev/null || true
}

run_is_active() {  # true if the run's recorded process is alive in this container
    local run=$1 state pid host
    state=$(status_get "$run" STATE)
    [ "$state" = running ] || return 1
    host=$(status_get "$run" HOST); pid=$(status_get "$run" PID)
    [ "$host" = "$(hostname)" ] || return 1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'gfx803-train' || return 1
    return 0
}

save_run_env() {  # write the run's training parameters
    local f; f="$(run_dir "$RUN_NAME")/run.env"
    {
        echo "# gfx803-train run settings, written $(date -Is)"
        local v
        for v in RUN_NAME MODEL_PATH VAE_PATH DATA_DIR STEPS SAVE_EVERY LR DIM ALPHA \
                 BATCH GRAD_ACCUM RESOLUTION MIN_BUCKET MAX_BUCKET WARMUP SCHEDULER INIT_WEIGHTS \
                 TRAIN_TE TE_LR; do
            printf '%s=%q\n' "$v" "${!v}"
        done
        declare -p EXTRA_ARGS
    } > "$f"
}

# Move a segment's files up into the run's output folder, renaming them to
# cumulative step numbers. Segment folders are named seg-<offset>-<steps>.
normalize_outputs() {
    local run=$1 od seg offset segsteps f name step newstep
    od=$(out_dir "$run")
    [ -d "$od" ] || return 0
    for seg in "$od"/seg-*; do
        [ -d "$seg" ] || continue
        [[ "$(basename "$seg")" =~ ^seg-([0-9]+)-([0-9]+)$ ]] || continue
        offset=${BASH_REMATCH[1]}; segsteps=${BASH_REMATCH[2]}
        for f in "$seg"/*; do
            [ -e "$f" ] || continue
            name=$(basename "$f")
            step=$(step_of "$name")
            if [ "$step" -ge 0 ]; then
                newstep=$((offset + step))
                if [ -d "$f" ]; then
                    mv -T "$f" "$od/$(printf '%s-step%08d-state' "$run" "$newstep")"
                else
                    mv -f "$f" "$od/$(printf '%s-step%08d.safetensors' "$run" "$newstep")"
                fi
            elif [ "$name" = "$run.safetensors" ]; then
                mv -f "$f" "$od/$name"
            elif [ "$name" = "$run-state" ] && [ -d "$f" ]; then
                mv -T "$f" "$od/$(printf '%s-step%08d-state' "$run" "$((offset + segsteps))")"
            else
                mv -f "$f" "$od/" 2>/dev/null || true
            fi
        done
        rmdir "$seg" 2>/dev/null || true
    done
}

state_is_valid() {  # a state folder that finished writing
    local d=$1
    compgen -G "$d/optimizer*.bin" > /dev/null || return 1
    compgen -G "$d/random_states_*.pkl" > /dev/null || return 1
    local o
    for o in "$d"/optimizer*.bin; do [ -s "$o" ] || return 1; done
    return 0
}

latest_state() {  # prints "<step> <dir>" of the newest valid state, or nothing
    local od best=-1 bestdir="" d s
    od=$(out_dir "$1")
    for d in "$od/$1"-step*-state; do
        [ -d "$d" ] || continue
        s=$(step_of "$(basename "$d")")
        if [ "$s" -gt "$best" ] && state_is_valid "$d"; then best=$s; bestdir=$d; fi
    done
    [ "$best" -ge 0 ] && echo "$best $bestdir"
}

latest_weights() {  # prints "<step> <file>" of the newest checkpoint, or nothing
    local od best=-1 bestf="" f s
    od=$(out_dir "$1")
    for f in "$od/$1"-step*.safetensors; do
        [ -s "$f" ] || continue
        s=$(step_of "$(basename "$f")")
        if [ "$s" -gt "$best" ]; then best=$s; bestf=$f; fi
    done
    [ "$best" -ge 0 ] && echo "$best $bestf"
}

run_progress() {  # prints the highest completed step we can resume from
    local st w
    st=$(latest_state "$1"); w=$(latest_weights "$1")
    local s1=${st%% *} s2=${w%% *}
    s1=${s1:--1}; s2=${s2:--1}
    [ "$s1" -gt "$s2" ] && echo "$s1" || echo "$s2"
}

list_runs() {
    [ -d "$RUNS_DIR" ] || { echo "No runs recorded yet."; return; }
    printf '%-52s %-12s %-12s %s\n' RUN STATE PROGRESS UPDATED
    local d run state total done upd
    for d in "$RUNS_DIR"/*/; do
        [ -f "$d/run.env" ] || continue
        run=$(basename "$d")
        state=$(status_get "$run" STATE)
        total=$(sed -n 's/^STEPS=//p' "$d/run.env")
        if run_is_active "$run"; then
            state="training"; done="?"   # don't touch files of a live run
        else
            normalize_outputs "$run"
            [ "$state" = running ] && state="died"
            done=$(run_progress "$run"); [ "$done" -lt 0 ] && done=0
            [ "$state" = completed ] && done=$total
        fi
        upd=$(status_get "$run" UPDATED)
        upd=$([ -n "$upd" ] && date -d "@$upd" '+%Y-%m-%d %H:%M' || echo -)
        printf '%-52s %-12s %-12s %s\n' "$run" "${state:-?}" "$done/$total" "$upd"
    done
}

find_unfinished_run() {  # newest run that stopped without completing
    local d run state best="" bestt=0 t
    [ -d "$RUNS_DIR" ] || return 0
    for d in "$RUNS_DIR"/*/; do
        [ -f "$d/run.env" ] || continue
        run=$(basename "$d")
        state=$(status_get "$run" STATE)
        case "$state" in
            running) run_is_active "$run" && continue ;;
            failed|interrupted|nan) ;;
            *) continue ;;
        esac
        t=$(status_get "$run" UPDATED); t=${t:-0}
        if [ "$t" -gt "$bestt" ]; then bestt=$t; best=$run; fi
    done
    echo "$best"
}

# Delete checkpoints and resume states whose weights contain NaN/Inf (or are
# unreadable), so a resume never starts from a poisoned state. Reads the
# safetensors format directly with numpy; no GPU involved.
purge_nan_checkpoints() {
    local od; od=$(out_dir "$1")
    [ -d "$od" ] || return 0
    python3 - "$od" "$1" <<'EOF'
import glob, json, os, shutil, struct, sys
import numpy as np
od, run = sys.argv[1], sys.argv[2]
DT = {"F32": np.float32, "F16": np.float16, "F64": np.float64}

def finite(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        buf = data[a:b]
        if len(buf) != b - a:
            return False
        if meta["dtype"] == "BF16":
            arr = (np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif meta["dtype"] in DT:
            arr = np.frombuffer(buf, DT[meta["dtype"]])
        else:
            continue
        if not np.isfinite(arr).all():
            return False
    return True

def ok(path):
    try:
        return finite(path)
    except Exception:
        return False

removed = []
for ck in sorted(glob.glob(os.path.join(od, run + "*.safetensors"))):
    if not ok(ck):
        removed.append(os.path.basename(ck)); os.remove(ck)
        state = ck[:-len(".safetensors")] + "-state"
        if os.path.isdir(state):
            shutil.rmtree(state); removed.append(os.path.basename(state))
for st in sorted(glob.glob(os.path.join(od, run + "*-state"))):
    if not os.path.isdir(st):
        continue
    weights = glob.glob(os.path.join(st, "*.safetensors"))
    if weights and not all(ok(w) for w in weights):
        shutil.rmtree(st); removed.append(os.path.basename(st))
for r in removed:
    print(f"  removed NaN/unreadable: {r}")
EOF
}

# ------------------------------------------------------------ GPU setup
setup_gpu_env() {  # fills GPU_ENV; applied only to the GPU commands, not this shell
    GPU_ENV=(-u HSA_DISABLE_FRAGMENT_ALLOCATOR -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_HIP_ALLOC_CONF
             -u GFX803_SGEMM_SHIM_DISABLE -u GFX803_SGEMM_SHIM_NO_BATCHED -u GFX803_SGEMM_SHIM_VERBOSE
             -u AMD_SERIALIZE_KERNEL -u HIP_LAUNCH_BLOCKING -u LD_PRELOAD
             -u GFX803_VRAM_RESERVE -u GFX803_VRAM_CAP
             GFX803_VRAM_RESERVE="$VRAM_RESERVE" GFX803_VRAM_CAP="$VRAM_CAP")
    local preload=() shim_path=""
    case "$RUNTIME" in
        patched) GPU_ENV+=(HSA_DISABLE_FRAGMENT_ALLOCATOR=0) ;;
        stock)
            [ -f "$STOCK_HSA" ] || die "stock libhsa-runtime64 not found: $STOCK_HSA"
            preload+=("$STOCK_HSA") ;;
        *) die "--runtime must be stock or patched" ;;
    esac
    case "$SHIM" in
        new) shim_path=$NEW_SHIM ;;
        original) shim_path=$ORIG_SHIM ;;
        none) shim_path="" ;;
        *) shim_path=$SHIM ;;
    esac
    if [ -n "$shim_path" ]; then
        [ -f "$shim_path" ] || die "shim not found: $shim_path"
        preload+=("$shim_path")
    fi
    [ "$SHIM" = original ] && warn "original shim: attention gradients are wrong; diagnostic use only"
    [ "$SHIM" = none ] && warn "no shim: stock rocBLAS fp32 GEMMs are unreliable on gfx803; diagnostic use only"
    [ "$NO_BATCHED" = 1 ] && GPU_ENV+=(GFX803_SGEMM_SHIM_NO_BATCHED=1)
    [ "$SHIM_VERBOSE" -gt 0 ] 2>/dev/null && GPU_ENV+=(GFX803_SGEMM_SHIM_VERBOSE="$SHIM_VERBOSE")
    if [ "$DEBUG_KERNELS" = 1 ]; then
        GPU_ENV+=(AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1)
        warn "--debug-kernels: much slower, and may hide timing-dependent bugs"
    fi
    PRELOAD_STR="${preload[*]}"
    [ -n "$PRELOAD_STR" ] && GPU_ENV+=(LD_PRELOAD="$PRELOAD_STR")
    EXPECT_SHIM=$shim_path
}

preflight_gpu() {
    log "Preflight: checking the GPU stack"
    if pgrep -f 'ComfyUI/main.py|main.py --listen' > /dev/null 2>&1; then
        warn "ComfyUI is running in this container; unload its models or stop it to free VRAM"
    fi
    env "${GPU_ENV[@]}" EXPECT_SHIM="$EXPECT_SHIM" VRAM_WARN_MIB="$VRAM_WARN_MIB" python3 - <<'EOF' || die "GPU preflight failed"
import os, sys
import torch
if not torch.cuda.is_available():
    sys.exit("torch cannot see the GPU")
free, total = torch.cuda.mem_get_info()
used_mib = (total - free) / 2**20
libs = set()
with open("/proc/self/maps") as maps:
    for line in maps:
        path = line.split()[-1]
        if any(s in path for s in ("libhsa-runtime64", "sgemm_shim", "librocblas.so")):
            libs.add(path)
print(f"  device: {torch.cuda.get_device_name(0)}, VRAM in use before training: {used_mib:.0f} MiB of {total / 2**20:.0f}")
for lib in sorted(libs):
    print(f"  loaded: {lib}")
expect = os.environ.get("EXPECT_SHIM", "")
if expect and not any(os.path.realpath(expect) == os.path.realpath(l) for l in libs):
    sys.exit(f"expected shim {expect} is not loaded")
if used_mib > float(os.environ.get("VRAM_WARN_MIB", "1024")):
    print(f"  WARNING: {used_mib:.0f} MiB of VRAM is already in use by something else "
          "(desktop session, ComfyUI...). Training with the desktop running on this GPU "
          "has hung the card; headless runs completed.", file=sys.stderr)
EOF
}

check_latents() {  # remove cached latents that can't be read or contain NaN/Inf
    python3 - "$DATA_DIR" <<'EOF'
import glob, os, sys
import numpy as np
bad = 0
files = glob.glob(os.path.join(sys.argv[1], "**", "*.npz"), recursive=True)
for f in files:
    try:
        with np.load(f) as z:
            ok = all(np.isfinite(z[k]).all() for k in z.files if z[k].dtype.kind == "f")
    except Exception:
        ok = False
    if not ok:
        os.remove(f)
        bad += 1
print(f"  cached latents: {len(files)} found, {bad} unreadable or non-finite removed")
EOF
}

prepare_latents() {
    local marker="$DATA_DIR/.gfx803_latents" want
    want="vae=$VAE_PATH model=$MODEL_PATH reso=$RESOLUTION buckets=$MIN_BUCKET-$MAX_BUCKET"
    if [ "$FRESH_LATENTS" = 1 ] || [ "$(cat "$marker" 2>/dev/null)" != "$want" ]; then
        local n
        n=$(find "$DATA_DIR" -name '*.npz' -type f | wc -l)
        [ "$n" -gt 0 ] && log "Removing $n cached latents (made with different settings or unknown origin)"
        find "$DATA_DIR" -name '*.npz' -type f -delete
        echo "$want" > "$marker"
    else
        check_latents
    fi
}

sd_scripts_has_arg() { grep -rqs -- "\"$1\"" "$SD_SCRIPTS/library" "$SD_SCRIPTS/train_network.py"; }

# ------------------------------------------------------------ arguments
# No arguments on a real terminal opens the menu, which hands back ordinary
# arguments; everything below treats them exactly like typed ones.
MENU=0
if [ $# -eq 0 ] && [ -t 0 ] && [ -t 1 ] && [ "${TERM:-dumb}" != dumb ] \
   && [ "${GFX803_TRAIN_NO_MENU:-0}" != 1 ]; then
    MENU=1
elif [ $# -eq 1 ] && [ "$1" = --menu ]; then
    MENU=1; shift
fi
if [ "$MENU" = 1 ]; then
    menu_out=$(mktemp)
    if ! DATA_ROOT="$DATA_ROOT" CKPT_DIR="$CKPT_DIR" VAE_DIR="$VAE_DIR" RUNS_DIR="$RUNS_DIR" \
         OUT_ROOT="$OUT_ROOT" DEFAULT_MODEL="$DEFAULT_MODEL" DEFAULT_VAE="$DEFAULT_VAE" \
         STEPS="$STEPS" LR="$LR" TE_LR="$TE_LR" DIM="$DIM" ALPHA="$ALPHA" SAVE_EVERY="$SAVE_EVERY" \
         RESOLUTION="$RESOLUTION" MIN_BUCKET="$MIN_BUCKET" MAX_BUCKET="$MAX_BUCKET" \
         python3 "$TRAINING_DIR/gfx803_train_menu.py" "$menu_out"; then
        rm -f "$menu_out"
        log "Cancelled."
        exit 0
    fi
    mapfile -d '' -t menu_args < "$menu_out"
    rm -f "$menu_out"
    set -- "${menu_args[@]}"
    log "Same as: gfx803-train $(printf '%q ' "$@")"
fi

while [ $# -gt 0 ]; do
    case "$1" in
        -d|--data) DATA_ARG=$2; shift 2 ;;
        -m|--model) MODEL_ARG=$2; shift 2 ;;
        --vae) VAE_ARG=$2; shift 2 ;;
        -n|--name) RUN_NAME=$(sanitize "$2"); shift 2 ;;
        --init-weights) INIT_WEIGHTS=$(readlink -f "$2"); shift 2 ;;
        -s|--steps) STEPS=$2; shift 2 ;;
        --save-every) SAVE_EVERY=$2; SAVE_EVERY_CLI=$2; shift 2 ;;
        --lr) LR=$2; shift 2 ;;
        --train-text-encoder) TRAIN_TE=1; shift ;;
        --te-lr) TE_LR=$2; shift 2 ;;
        --dim) DIM=$2; shift 2 ;;
        --alpha) ALPHA=$2; shift 2 ;;
        --batch) BATCH=$2; shift 2 ;;
        --grad-accum) GRAD_ACCUM=$2; shift 2 ;;
        --resolution) RESOLUTION=$2; shift 2 ;;
        --buckets) MIN_BUCKET=${2%%:*}; MAX_BUCKET=${2##*:}; shift 2 ;;
        --warmup) WARMUP=$2; shift 2 ;;
        --scheduler) SCHEDULER=$2; shift 2 ;;
        --fresh-latents) FRESH_LATENTS=1; shift ;;
        --runtime) RUNTIME=$2; DIAG_ARGS+=("$1" "$2"); shift 2 ;;
        --shim) SHIM=$2; DIAG_ARGS+=("$1" "$2"); shift 2 ;;
        --no-batched) NO_BATCHED=1; DIAG_ARGS+=("$1"); shift ;;
        --debug-kernels) DEBUG_KERNELS=1; DIAG_ARGS+=("$1"); shift ;;
        --shim-verbose) SHIM_VERBOSE=$2; DIAG_ARGS+=("$1" "$2"); shift 2 ;;
        --vram-reserve) VRAM_RESERVE=$2; DIAG_ARGS+=("$1" "$2"); shift 2 ;;
        --vram-cap) VRAM_CAP=$2; DIAG_ARGS+=("$1" "$2"); shift 2 ;;
        --no-preflight) PREFLIGHT=0; DIAG_ARGS+=("$1"); shift ;;
        --nan-retries) NAN_RETRIES=$2; shift 2 ;;
        --resume) RESUME_MODE=$2; shift 2 ;;
        --resume-run) RESUME_RUN=$2; shift 2 ;;
        -y|--yes) ASSUME_YES=1; shift ;;
        --list-models) ACTION=list-models; shift ;;
        --list-data) ACTION=list-data; shift ;;
        --list-runs) ACTION=list-runs; shift ;;
        --print-defaults) ACTION=print-defaults; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done
case "$RESUME_MODE" in ask|auto|never) ;; *) die "--resume must be ask, auto or never" ;; esac
[[ "$VRAM_RESERVE" == auto || "$VRAM_RESERVE" =~ ^[0-9]+$ ]] \
    || die "--vram-reserve must be auto or a whole number of MiB"
[[ -z "$VRAM_CAP" || "$VRAM_CAP" == none || "$VRAM_CAP" =~ ^[0-9]+$ ]] \
    || die "--vram-cap must be none or a whole number of MiB"
for v in STEPS SAVE_EVERY DIM BATCH GRAD_ACCUM RESOLUTION MIN_BUCKET MAX_BUCKET WARMUP NAN_RETRIES; do
    [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v must be a whole number (got '${!v}')"
done
for v in LR TE_LR; do
    [[ "${!v}" =~ ^[0-9]*\.?[0-9]+([eE]-?[0-9]+)?$ ]] || die "$v must be a number such as 1e-4 (got '${!v}')"
done

case "$ACTION" in
    list-models) list_models; exit 0 ;;
    list-data) list_data; exit 0 ;;
    list-runs) list_runs; exit 0 ;;
    print-defaults)
        for v in DATA_ROOT CKPT_DIR VAE_DIR RUNS_DIR OUT_ROOT DEFAULT_MODEL DEFAULT_VAE STEPS LR TE_LR DIM \
                 ALPHA SAVE_EVERY RESOLUTION MIN_BUCKET MAX_BUCKET; do
            printf '%s=%s\n' "$v" "${!v}"
        done
        exit 0 ;;
esac

mkdir -p "$RUNS_DIR" "$OUT_ROOT"

# Refuse to start a second training on the same GPU.
for d in "$RUNS_DIR"/*/; do
    [ -d "$d" ] || continue
    r=$(basename "$d")
    if run_is_active "$r"; then
        die "run '$r' is still training in this container (PID $(status_get "$r" PID))"
    fi
done

# ------------------------------------------------------------ new or resume?
RESUMING=0
if [ -n "$RESUME_RUN" ]; then
    [ -f "$(run_dir "$RESUME_RUN")/run.env" ] || die "no recorded run named '$RESUME_RUN' (see --list-runs)"
    RESUMING=1; CANDIDATE=$RESUME_RUN
elif [ "$RESUME_MODE" != never ] && [ -z "$RUN_NAME" ]; then
    CANDIDATE=$(find_unfinished_run)
    if [ -n "$CANDIDATE" ]; then
        normalize_outputs "$CANDIDATE"
        c_total=$(sed -n 's/^STEPS=//p' "$(run_dir "$CANDIDATE")/run.env")
        c_done=$(run_progress "$CANDIDATE"); [ "$c_done" -lt 0 ] && c_done=0
        c_state=$(status_get "$CANDIDATE" STATE)
        [ "$c_state" = running ] && c_state="stopped unexpectedly (GPU hang or reboot?)"
        log "Unfinished run found: $CANDIDATE"
        log "  status: $c_state; resumable from step $c_done of $c_total"
        if [ "$RESUME_MODE" = auto ]; then
            RESUMING=1
        elif [ "$ASSUME_YES" = 1 ] || [ ! -t 0 ]; then
            RESUMING=1
        else
            read -r -p "[gfx803-train] Resume it? [Y]es / [n]o, start a new run / [a]bandon it (stop asking) " ans
            case "$ans" in
                ""|[Yy]*) RESUMING=1 ;;
                [Aa]*) status_set "$CANDIDATE" STATE=abandoned; log "Marked $CANDIDATE as abandoned." ;;
                *) log "Starting a new run; $CANDIDATE is left as it is." ;;
            esac
        fi
    fi
fi

SEG_OFFSET=0
RESUME_FLAGS=()
if [ "$RESUMING" = 1 ]; then
    # Reload the run's own training settings; diagnostics stay as given now.
    CLI_EXTRA=("${EXTRA_ARGS[@]}")
    # Runs recorded before the text encoder option existed trained the UNet only.
    TRAIN_TE=0; TE_LR=5e-5
    # shellcheck disable=SC1090
    source "$(run_dir "$CANDIDATE")/run.env"
    [ -d "$DATA_DIR" ] || die "run $RUN_NAME's dataset folder no longer exists: $DATA_DIR
    (if it moved, fix DATA_DIR= in $(run_dir "$RUN_NAME")/run.env)"
    if [ -n "$SAVE_EVERY_CLI" ] && [ "$SAVE_EVERY_CLI" != "$SAVE_EVERY" ]; then
        log "Checkpoint interval changed from $SAVE_EVERY to $SAVE_EVERY_CLI steps for this run"
        SAVE_EVERY=$SAVE_EVERY_CLI
        save_run_env
    fi
    [ ${#CLI_EXTRA[@]} -gt 0 ] && warn "extra train_network.py args are ignored when resuming"
    normalize_outputs "$RUN_NAME"
    purge_nan_checkpoints "$RUN_NAME"
    st=$(latest_state "$RUN_NAME")
    wt=$(latest_weights "$RUN_NAME")
    if [ -n "$st" ]; then
        SEG_OFFSET=${st%% *}
        RESUME_FLAGS+=(--resume="${st#* }")
        # sd-scripts may restore its step counter from the state; pin it to 0 so
        # this segment's numbering starts fresh (the wrapper adds the offset).
        sd_scripts_has_arg --initial_step && RESUME_FLAGS+=(--initial_step=0)
        case "$SCHEDULER" in
            constant|constant_with_warmup) ;;
            *) warn "scheduler '$SCHEDULER' decays over the run; after a resume its schedule is approximate" ;;
        esac
        log "Resuming $RUN_NAME from full state at step $SEG_OFFSET"
    elif [ -n "$wt" ]; then
        SEG_OFFSET=${wt%% *}
        RESUME_FLAGS+=(--network_weights="${wt#* }")
        log "Resuming $RUN_NAME from LoRA weights at step $SEG_OFFSET (no saved state: optimizer restarts)"
    else
        log "No checkpoint saved yet for $RUN_NAME; restarting it from step 0"
        [ -n "${INIT_WEIGHTS:-}" ] && RESUME_FLAGS+=(--network_weights="$INIT_WEIGHTS")
    fi
    if [ "$SEG_OFFSET" -ge "$STEPS" ]; then
        log "All $STEPS steps are already done; marking the run completed."
        last=$(latest_weights "$RUN_NAME")
        [ -f "$(out_dir "$RUN_NAME")/$RUN_NAME.safetensors" ] || cp "${last#* }" "$(out_dir "$RUN_NAME")/$RUN_NAME.safetensors"
        status_set "$RUN_NAME" STATE=completed
        exit 0
    fi
else
    MODEL_PATH=$(find_model "$CKPT_DIR" "${MODEL_ARG:-$DEFAULT_MODEL}" checkpoint) || exit 1
    VAE_Q=${VAE_ARG:-$DEFAULT_VAE}
    if [ "$VAE_Q" = none ]; then VAE_PATH=""; else VAE_PATH=$(find_model "$VAE_DIR" "$VAE_Q" VAE) || exit 1; fi
    DATA_DIR=$(find_data "${DATA_ARG:-$DEFAULT_DATA}") || exit 1
    compgen -G "$DATA_DIR/[0-9]*_*" > /dev/null \
        || warn "no <repeats>_<concept> subfolders in $DATA_DIR; sd-scripts expects e.g. 10_character/"
    if [ -z "$RUN_NAME" ]; then
        ds=$(basename "$DATA_DIR"); ds=${ds%_training_data}; ds=${ds%_data}; ds=${ds%-data}
        mb=$(basename "$MODEL_PATH"); mb=${mb%.*}
        RUN_NAME=$(sanitize "${ds}_${mb}_d${DIM}_$(date +%Y%m%d-%H%M)")
        base=$RUN_NAME; i=2
        while [ -e "$(run_dir "$RUN_NAME")" ]; do RUN_NAME="${base}-$i"; i=$((i + 1)); done
    fi
    [ -e "$(run_dir "$RUN_NAME")" ] && die "a run named '$RUN_NAME' already exists (use --resume-run to continue it)"
    if [ -n "$INIT_WEIGHTS" ]; then
        [ -f "$INIT_WEIGHTS" ] || die "--init-weights file not found: $INIT_WEIGHTS"
        RESUME_FLAGS+=(--network_weights="$INIT_WEIGHTS")
        log "Starting from existing LoRA weights: $INIT_WEIGHTS (optimizer starts fresh)"
    fi
fi

SEG_STEPS=$((STEPS - SEG_OFFSET))
SEG_DIR="$(out_dir "$RUN_NAME")/seg-$SEG_OFFSET-$SEG_STEPS"
LOG_FILE="$(run_dir "$RUN_NAME")/train.log"

setup_gpu_env

CMD=(env "${GPU_ENV[@]}" "$ACCELERATE" launch --num_processes=1 --num_machines=1 --mixed_precision=no --dynamo_backend=no
    "$TRAINING_DIR/gfx803_vram_cap.py" "$SD_SCRIPTS/train_network.py"
    --pretrained_model_name_or_path="$MODEL_PATH"
    --train_data_dir="$DATA_DIR"
    --output_dir="$SEG_DIR"
    --output_name="$RUN_NAME"
    --save_model_as=safetensors --save_precision=float
    --resolution="$RESOLUTION,$RESOLUTION"
    --enable_bucket --min_bucket_reso="$MIN_BUCKET" --max_bucket_reso="$MAX_BUCKET"
    --cache_latents --cache_latents_to_disk
    --mixed_precision=no --no_half_vae
    --optimizer_type=AdamW
    --learning_rate="$LR" --unet_lr="$LR"
    --lr_scheduler="$SCHEDULER" --lr_warmup_steps="$WARMUP"
    --max_grad_norm=1.0
    --max_train_steps="$SEG_STEPS"
    --save_every_n_steps="$SAVE_EVERY" --save_state
    --train_batch_size="$BATCH" --gradient_accumulation_steps="$GRAD_ACCUM"
    --gradient_checkpointing
    --network_module=networks.lora
    --network_dim="$DIM" --network_alpha="$ALPHA"
    --caption_extension=.txt --shuffle_caption --keep_tokens=1 --max_token_length=225
    --max_data_loader_n_workers=0
    --logging_dir="$(run_dir "$RUN_NAME")/tb" --log_with=tensorboard)
[ -n "$VAE_PATH" ] && CMD+=(--vae="$VAE_PATH")
if [ "$TRAIN_TE" = 1 ]; then
    CMD+=(--text_encoder_lr="$TE_LR")
else
    CMD+=(--network_train_unet_only)
fi
CMD+=("${RESUME_FLAGS[@]}" "${EXTRA_ARGS[@]}")

SUMMARY="run=$RUN_NAME steps=$SEG_OFFSET->$STEPS runtime=$RUNTIME shim=$SHIM no_batched=$NO_BATCHED text_encoder=$([ "$TRAIN_TE" = 1 ] && echo "trained@$TE_LR" || echo frozen) vram_reserve=$VRAM_RESERVE vram_cap=${VRAM_CAP:-<from reserve>} LD_PRELOAD=${PRELOAD_STR:-<none>}"

if [ "$DRY_RUN" = 1 ]; then
    log "Dry run: $SUMMARY"
    log "Model: $MODEL_PATH"
    log "VAE:   ${VAE_PATH:-<from checkpoint>}"
    log "Data:  $DATA_DIR"
    printf '  %q' "${CMD[@]}"; echo
    exit 0
fi

# ------------------------------------------------------------ run it
mkdir -p "$(run_dir "$RUN_NAME")" "$SEG_DIR"
[ "$RESUMING" = 0 ] && save_run_env
if [ "$PREFLIGHT" = 1 ]; then preflight_gpu 2>&1 | tee -a "$LOG_FILE"; [ "${PIPESTATUS[0]}" = 0 ] || exit 1; fi
if [ "$RESUMING" = 0 ]; then prepare_latents | tee -a "$LOG_FILE"; else check_latents | tee -a "$LOG_FILE"; fi

status_set "$RUN_NAME" STATE=running PID=$$ HOST="$(hostname)" STARTED="$(date +%s)" \
    SEGMENT="$SEG_OFFSET-$SEG_STEPS" RUNTIME="$RUNTIME" SHIM="$SHIM"

TPID=""
stop_training() {  # stop the whole training process group
    [ -n "$TPID" ] || return 0
    kill -TERM -- "-$TPID" 2>/dev/null
    local i
    for i in $(seq 20); do kill -0 "$TPID" 2>/dev/null || return 0; sleep 1; done
    kill -KILL -- "-$TPID" 2>/dev/null
}
on_signal() {
    stop_training
    status_set "$RUN_NAME" STATE=interrupted
    normalize_outputs "$RUN_NAME"
    log "Interrupted. Run again to resume $RUN_NAME."
    exit 130
}
trap on_signal INT TERM

{ echo "===== $(date -Is) $SUMMARY"; printf '%q ' "${CMD[@]}"; echo; } >> "$LOG_FILE"
log "Starting: $SUMMARY"
log "Log: $LOG_FILE"
LOG_START=$(stat -c %s "$LOG_FILE")

# Training runs in its own process group so it can be stopped as a whole;
# meanwhile the loss is watched for NaN/Inf.
setsid "${CMD[@]}" > >(tee -a "$LOG_FILE") 2>&1 < /dev/null &
TPID=$!
NAN_HIT=0
while kill -0 "$TPID" 2>/dev/null; do
    sleep 5
    if tail -c +"$((LOG_START + 1))" "$LOG_FILE" | tail -c 20000 | tr '\r' '\n' \
            | grep -qE 'avr_loss=(nan|inf)'; then
        NAN_HIT=1
        echo; log "Loss became NaN; stopping this segment."
        stop_training
        break
    fi
done
wait "$TPID"; RC=$?
TPID=""
sleep 1   # let tee finish writing
if [ "$NAN_HIT" = 0 ] && tail -c +"$((LOG_START + 1))" "$LOG_FILE" | tail -c 20000 | tr '\r' '\n' \
        | grep -qE 'avr_loss=(nan|inf)'; then
    NAN_HIT=1; log "Loss became NaN near the end of this segment."
fi
trap - INT TERM

normalize_outputs "$RUN_NAME"
FINAL="$(out_dir "$RUN_NAME")/$RUN_NAME.safetensors"
if [ "$NAN_HIT" = 1 ]; then
    purge_nan_checkpoints "$RUN_NAME" | tee -a "$LOG_FILE"
    status_set "$RUN_NAME" STATE=nan EXIT_CODE=nan
    if [ "$NAN_RETRIES" -gt 0 ]; then
        log "Resuming from step $(run_progress "$RUN_NAME") (NaN retries left after this: $((NAN_RETRIES - 1)))"
        exec env GFX803_NAN_RETRIES=$((NAN_RETRIES - 1)) bash "$0" --resume-run "$RUN_NAME" \
            "${DIAG_ARGS[@]}"
    fi
    status_set "$RUN_NAME" STATE=failed
    log "NaN retries used up. Clean checkpoints are kept; the latest is step $(run_progress "$RUN_NAME")."
    exit 1
elif [ "$RC" = 0 ] && [ -s "$FINAL" ]; then
    purge_nan_checkpoints "$RUN_NAME" | tee -a "$LOG_FILE"
    if [ -s "$FINAL" ]; then
        status_set "$RUN_NAME" STATE=completed EXIT_CODE=0
        log "Completed $RUN_NAME. Files in $(out_dir "$RUN_NAME"):"
        ls -1 "$(out_dir "$RUN_NAME")" | grep -v -- '-state$' | sed 's/^/    /'
    else
        status_set "$RUN_NAME" STATE=failed EXIT_CODE=nan
        log "The final LoRA contained NaN and was removed. Run the script again to resume."
        exit 1
    fi
else
    status_set "$RUN_NAME" STATE=failed EXIT_CODE="$RC"
    log "Training exited with code $RC. Run the script again to resume from step $(run_progress "$RUN_NAME")."
    exit "$RC"
fi
