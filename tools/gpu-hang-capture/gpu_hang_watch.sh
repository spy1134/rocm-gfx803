#!/bin/bash
# Host-side GPU hang capture for a gfx803 (Polaris) card. Run as root in its own terminal
# (or screen window) before training:
#   sudo bash gpu_hang_watch.sh [output_dir]        # default /var/log/gpu_hang
#
# Triggers a capture when any of these happens:
#   1. the kernel logs a GPU timeout / failed queue preemption / reset;
#   2. "no progress": a compute queue has work pending or a dispatch in
#      flight, and neither its read pointer nor its end-of-pipe pointer has
#      moved for PROGRESS_SECONDS. Kernels here finish in milliseconds, so a
#      frozen pointer is a hang regardless of what power or clocks show;
#   3. "stall" (off unless STALL_WATTS is set): 100% busy below STALL_WATTS
#      for STALL_SECONDS after real load. Power is a poor signal once clocks
#      or power are capped, because healthy training then dips below any
#      threshold a hang would cross.
#
# The windows must stay well under amdgpu's 10 s ring timeout: with a desktop
# running, the gfx ring times out and the GPU is reset 10 s into a hang, and
# after the reset the stuck compute queue is gone. The host itself has frozen
# within 5 s of a hang, so the queue is polled every second and the most
# valuable evidence is written first. Every capture step is read-only, so a
# false capture costs nothing.
#
# Each capture (in <output_dir>/<timestamp>/) contains:
#   trigger.txt        why it fired
#   kernel.txt         kernel messages seen by this watcher (not lost to journald)
#   dmesg.txt          full kernel ring buffer
#   kfd_hqds/rls       KFD queue state (read/write pointers, runlist)
#   hang_decode.txt    the stuck queue's dispatch packets, launch shape and
#                      disassembly of the kernel it is stuck on
#                      (needs gpu_hang_decode.py next to this script)
#   fence_info.txt     amdgpu fence state per ring
#   kfd_users.txt      processes holding /dev/kfd
#   umr_*.txt          wave dumps, only with UMR_WAVES=1
#   sensors.txt        busy/power/temperature at capture time
if [ -z "${PCI:-}" ]; then
    for d in /sys/bus/pci/devices/*; do
        if [ "$(cat "$d/vendor" 2>/dev/null)" = 0x1002 ] && [ -e "$d/drm" ]; then
            PCI=$(basename "$d"); break
        fi
    done
fi
[ -n "${PCI:-}" ] || { echo "No AMD GPU found; set PCI=<domain:bus:device.function>" >&2; exit 1; }
OUT=${1:-/var/log/gpu_hang}
PROGRESS_SECONDS=${PROGRESS_SECONDS:-4}
STALL_WATTS=${STALL_WATTS:-0}
STALL_SECONDS=${STALL_SECONDS:-4}
LOAD_WATTS=${LOAD_WATTS:-145}
DEV=/sys/bus/pci/devices/$PCI
DRI=$(basename "$(ls -d "$DEV"/drm/card* 2>/dev/null | head -n 1)" | sed 's/card//')
DBG=/sys/kernel/debug/dri/${DRI:-1}
KFD_DBG=/sys/kernel/debug/kfd
DECODER=$(dirname "$(readlink -f "$0")")/gpu_hang_decode.py
HW=$(ls -d "$DEV"/hwmon/hwmon* 2>/dev/null | head -n 1)
PWR=$HW/power1_average; [ -r "$PWR" ] || PWR=$HW/power1_input

[ "$(id -u)" = 0 ] || { echo "run as root (sudo)"; exit 1; }
mkdir -p "$OUT"
KMSG=$(mktemp)
journalctl -k -f -n 0 -o short-precise > "$KMSG" 2>/dev/null &
JPID=$!
trap 'kill $JPID 2>/dev/null; rm -f "$KMSG"; exit 0' INT TERM EXIT
KERNEL_PATTERN='ring .*timeout|qcm fence|preemption|Failed to evict|GPU reset|hung|HW Exception'
stall_rule="off"
[ "$STALL_WATTS" -gt 0 ] && stall_rule="busy 100% and < ${STALL_WATTS} W for ${STALL_SECONDS}s after load"
echo "Watching $PCI (debugfs $DBG). No-progress rule: queue pointers frozen for ${PROGRESS_SECONDS}s. Power stall rule: $stall_rule. Captures -> $OUT"

watts() { awk '{ printf "%d", $1 / 1000000 }' "$PWR" 2>/dev/null || echo 0; }
busy() { cat "$DEV/gpu_busy_percent" 2>/dev/null || echo 0; }

capture() {
    local reason=$1 d
    d="$OUT/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$d"
    echo "$reason" > "$d/trigger.txt"
    echo "busy=$(busy)% power=$(watts)W temp=$(awk '{print $1/1000}' "$HW/temp1_input" 2>/dev/null)C" > "$d/sensors.txt"
    cp "$KMSG" "$d/kernel.txt"
    # Most valuable first: the machine may lock up within seconds.
    cat "$KFD_DBG/hqds" > "$d/kfd_hqds.txt" 2>&1; sync
    if [ -f "$DECODER" ] && command -v python3 >/dev/null 2>&1; then
        timeout 90 python3 "$DECODER" --outdir "$d" > "$d/hang_decode.txt" 2>&1; sync
    fi
    # kfd/mqds is left alone: reading it with queues active oopses this kernel
    # (NULL call in pqm_debugfs_mqds) and turns a GPU hang into a host lockup.
    cat "$KFD_DBG/rls" > "$d/kfd_rls.txt" 2>&1
    cat "$DBG/amdgpu_fence_info" > "$d/fence_info.txt" 2>&1
    fuser -v /dev/kfd > "$d/kfd_users.txt" 2>&1
    ps -eo pid,ppid,etimes,stat,cmd --sort=pid | grep -iE 'python|accelerate|comfy|train' | grep -v grep >> "$d/kfd_users.txt"
    cp "$DEV/devcoredump/data" "$d/devcoredump.txt" 2>/dev/null
    sync
    # Halting waves can hang a GPU that is still healthy, and a stall capture
    # may be a false alarm, so wave dumps are opt-in.
    if [ "${UMR_WAVES:-0}" = 1 ] && command -v umr >/dev/null 2>&1; then
        # -wa needs an explicit ring on this umr build; the wave list itself
        # covers every active wave (compute included), the ring is only used
        # to look up shader addresses for disassembly.
        timeout 60 umr -O halt_waves,bits -wa gfx > "$d/umr_waves_1.txt" 2>&1
        sleep 2
        timeout 60 umr -O halt_waves -wa gfx      > "$d/umr_waves_2.txt" 2>&1
    fi
    dmesg > "$d/dmesg.txt"
    sync
    echo "$(date '+%H:%M:%S') Captured GPU hang evidence in $d ($reason)"
}

queue_state() { python3 "$DECODER" --state 2>/dev/null; }

loaded_at=0
low_since=0
last_state=""
state_since=0
while true; do
    now=$(date +%s)
    if grep -qiE "$KERNEL_PATTERN" "$KMSG"; then
        capture "kernel: $(grep -iE "$KERNEL_PATTERN" "$KMSG" | head -n 1)"
        sleep 120; : > "$KMSG"; low_since=0; loaded_at=0; last_state=""; continue
    fi

    state=$(queue_state)
    if [ -z "$state" ] || [ "$state" != "$last_state" ]; then
        last_state=$state
        state_since=$now
    elif [ $((now - state_since)) -ge "$PROGRESS_SECONDS" ]; then
        capture "no progress for $((now - state_since))s: $(echo "$state" | tr '\n' ' ')"
        # A hang that persists is captured again after the pause, which shows
        # whether the same dispatch is still stuck.
        sleep 10; last_state=""; continue
    fi

    if [ "$STALL_WATTS" -gt 0 ]; then
        w=$(watts); b=$(busy)
        [ "$w" -ge "$LOAD_WATTS" ] && loaded_at=$now
        if [ "$b" -ge 99 ] && [ "$w" -lt "$STALL_WATTS" ] && [ $((now - loaded_at)) -lt 120 ]; then
            [ "$low_since" = 0 ] && low_since=$now
            if [ $((now - low_since)) -ge "$STALL_SECONDS" ]; then
                capture "stall: busy ${b}% at ${w} W for $((now - low_since))s after load"
                sleep 10; low_since=0; continue
            fi
        else
            low_since=0
        fi
    fi
    sleep 1
done
