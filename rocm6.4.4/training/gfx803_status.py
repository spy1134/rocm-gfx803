"""Readings for gfx803-app: GPU (sysfs), system (psutil), training runs.

Everything here only reads files; nothing touches the GPU through HIP, so
the app can watch a training run without competing with it.
"""
import glob
import os
import re
import time

import gfx803_queue as gq

OUT_ROOT = os.path.join(gq.WORKSPACE, "output")


# ------------------------------------------------------------------ GPU
def gpu_device():
    """The first AMD card's sysfs device folder, or None."""
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        if "-" in os.path.basename(card):
            continue
        try:
            with open(os.path.join(card, "device", "vendor")) as f:
                if f.read().strip() == "0x1002":
                    return os.path.join(card, "device")
        except OSError:
            pass
    return None


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _current_level(path):
    """The active entry of a pp_dpm_* table, in MHz."""
    try:
        with open(path) as f:
            for line in f:
                if "*" in line:
                    m = re.search(r"(\d+)\s*Mhz", line, re.I)
                    return int(m.group(1)) if m else None
    except OSError:
        pass
    return None


def gpu_sample(dev):
    """Dict of GPU readings; missing sensors are None."""
    if not dev:
        return {}
    hw = (glob.glob(os.path.join(dev, "hwmon", "hwmon*")) or [""])[0]
    power = _read_int(os.path.join(hw, "power1_average"))
    if power is None:
        power = _read_int(os.path.join(hw, "power1_input"))
    temp = _read_int(os.path.join(hw, "temp1_input"))
    used = _read_int(os.path.join(dev, "mem_info_vram_used"))
    total = _read_int(os.path.join(dev, "mem_info_vram_total"))
    return {
        "busy": _read_int(os.path.join(dev, "gpu_busy_percent")),
        "power_w": power / 1e6 if power is not None else None,
        "power_cap_w": (_read_int(os.path.join(hw, "power1_cap")) or 0) / 1e6 or None,
        "temp_c": temp / 1000 if temp is not None else None,
        "fan_rpm": _read_int(os.path.join(hw, "fan1_input")),
        "sclk_mhz": _current_level(os.path.join(dev, "pp_dpm_sclk")),
        "mclk_mhz": _current_level(os.path.join(dev, "pp_dpm_mclk")),
        "vram_used_gib": used / 2**30 if used is not None else None,
        "vram_total_gib": total / 2**30 if total else None,
    }


# ------------------------------------------------------------------ system
def system_sample(psutil):
    """CPU, memory, disk and network readings via psutil (host-wide with
    --pid host / --network host)."""
    vm = psutil.virtual_memory()
    net = psutil.net_io_counters()
    disk = psutil.disk_io_counters()
    try:
        ws = psutil.disk_usage(gq.WORKSPACE)
    except OSError:
        ws = None
    temps = {}
    try:
        for name, entries in psutil.sensors_temperatures().items():
            if entries:
                temps[name] = entries[0].current
    except (AttributeError, OSError):
        pass
    return {
        "time": time.time(),
        "cpu": psutil.cpu_percent(),
        "cpu_per_core": psutil.cpu_percent(percpu=True),
        "load": os.getloadavg(),
        "ram_used_gib": (vm.total - vm.available) / 2**30,
        "ram_total_gib": vm.total / 2**30,
        "swap_used_gib": psutil.swap_memory().used / 2**30,
        "net_rx": net.bytes_recv, "net_tx": net.bytes_sent,
        "disk_read": disk.read_bytes if disk else 0, "disk_write": disk.write_bytes if disk else 0,
        "ws_used_gib": ws.used / 2**30 if ws else None, "ws_total_gib": ws.total / 2**30 if ws else None,
        "cpu_temp_c": temps.get("k10temp") or temps.get("coretemp"),
    }


# ------------------------------------------------------------------ training
def tail(path, nbytes=65536):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(f.tell() - nbytes, 0))
            return f.read().decode(errors="replace").replace("\r", "\n")
    except OSError:
        return ""


# Only sd-scripts' training bar ("steps: 35%|...| 420/1200 [...]"): the
# latent-caching bar printed before training has the same shape.
PROGRESS = re.compile(r"steps:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.]+)(s/it|it/s)"
                      r"(?:,\s*avr_loss=([\d.]+|nan|inf))?")


def run_progress(run):
    """Dict with step, total, s_per_step, loss for a run, or None before
    training has started. Steps count across resumed segments."""
    if not run:
        return None
    m = None
    for m in PROGRESS.finditer(tail(os.path.join(gq.RUNS_DIR, run, "train.log"))):
        pass
    if not m:
        return None
    n, seg_total, rate, unit, loss = int(m.group(1)), int(m.group(2)), float(m.group(5)), m.group(6), m.group(7)
    sps = rate if unit == "s/it" else (1 / rate if rate else None)
    offset, total = 0, seg_total
    seg = gq.read_kv(os.path.join(gq.RUNS_DIR, run, "status")).get("SEGMENT", "")
    # SEGMENT is <offset>-<length>: a run resumed at step 100 of 1200 trains
    # segment 100-1100 and its bar counts 0..1100.
    if re.fullmatch(r"\d+-\d+", seg):
        offset, length = (int(x) for x in seg.split("-"))
        total = offset + length
    return {"step": offset + n, "total": total, "s_per_step": sps, "loss": loss}


def checkpoints(run):
    """[(step, path)] of a run's LoRA checkpoints, final file as step None."""
    out = []
    d = os.path.join(OUT_ROOT, run)
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        m = re.search(r"-step(\d{8})\.safetensors$", f)
        out.append((int(m.group(1)) if m else None, f))
    return out


def runs():
    """Recorded runs, newest first, as dicts."""
    out = []
    for d in glob.glob(os.path.join(gq.RUNS_DIR, "*")):
        env = gq.read_kv(os.path.join(d, "run.env"))
        if not env:
            continue
        status = gq.read_kv(os.path.join(d, "status"))
        name = os.path.basename(d)
        prog = run_progress(name)
        out.append({
            "name": name,
            "state": status.get("STATE", "?"),
            "steps": int(env.get("STEPS", "0") or 0),
            "step": prog["step"] if prog else None,
            "loss": prog["loss"] if prog else None,
            "updated": int(status.get("UPDATED", "0") or 0),
            "data": os.path.basename(env.get("DATA_DIR", "").rstrip("/")),
            "model": os.path.splitext(os.path.basename(env.get("MODEL_PATH", "")))[0],
        })
    out.sort(key=lambda r: r["updated"], reverse=True)
    return out
