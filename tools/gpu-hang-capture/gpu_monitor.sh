#!/bin/bash
# Host-side GPU logger for a gfx803 (Polaris) card: temperature, power, fan, clocks, load
# and VRAM every few seconds, synced to disk after each line so the log
# survives a hard hang up to its last few seconds.
#
# Usage (on the host, not in the container):
#   bash gpu_monitor.sh [logfile]          # default ~/gpu_monitor.log
#   PCI=0000:03:00.0 INTERVAL=2 bash gpu_monitor.sh   # pick a card; default: first AMD GPU
# Stop with Ctrl+C.
if [ -z "${PCI:-}" ]; then
    for d in /sys/bus/pci/devices/*; do
        if [ "$(cat "$d/vendor" 2>/dev/null)" = 0x1002 ] && [ -e "$d/drm" ]; then
            PCI=$(basename "$d"); break
        fi
    done
fi
[ -n "${PCI:-}" ] || { echo "No AMD GPU found; set PCI=<domain:bus:device.function>" >&2; exit 1; }
INTERVAL=${INTERVAL:-2}
OUT=${1:-$HOME/gpu_monitor.log}
DEV=/sys/bus/pci/devices/$PCI

HW=""
for d in "$DEV"/hwmon/hwmon*; do [ -d "$d" ] && HW=$d && break; done
[ -n "$HW" ] || { echo "No hwmon directory found for $PCI" >&2; exit 1; }

rd() { local v; v=$(cat "$1" 2>/dev/null) && echo "$v" || echo NA; }
scale() { awk -v v="$1" -v d="$2" 'BEGIN { if (v == "NA") print "NA"; else printf "%.1f", v / d }'; }

POWER_FILE=$HW/power1_average
[ -r "$POWER_FILE" ] || POWER_FILE=$HW/power1_input

echo "Logging $PCI ($HW) every ${INTERVAL}s to $OUT"
printf '%s\n' "time temp_C power_W fan_rpm pwm sclk_MHz mclk_MHz busy_pct vram_used_MiB" >> "$OUT"
while true; do
    line=$(printf '%s %s %s %s %s %s %s %s %s' \
        "$(date +%H:%M:%S)" \
        "$(scale "$(rd "$HW/temp1_input")" 1000)" \
        "$(scale "$(rd "$POWER_FILE")" 1000000)" \
        "$(rd "$HW/fan1_input")" \
        "$(rd "$HW/pwm1")" \
        "$(scale "$(rd "$HW/freq1_input")" 1000000)" \
        "$(scale "$(rd "$HW/freq2_input")" 1000000)" \
        "$(rd "$DEV/gpu_busy_percent")" \
        "$(scale "$(rd "$DEV/mem_info_vram_used")" 1048576)")
    echo "$line"
    echo "$line" >> "$OUT"
    sync "$OUT" 2>/dev/null
    sleep "$INTERVAL"
done
