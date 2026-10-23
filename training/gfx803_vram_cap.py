"""Runs a training script with PyTorch's share of VRAM capped.

Usage: gfx803_vram_cap.py SCRIPT [args...]

Uncapped, PyTorch's caching allocator keeps what it has freed and grows
until the card is full, whatever the training needs: an SD1.5 LoRA run that
needs about 6.1 GB took 7.99 of an RX 590's 8 GB. With a desktop on the same
card, amdgpu then moved buffers out to system memory and back, the GPU
waited on PCIe (94% busy at 60-70 W, steps 15x slower), and the compositor
could no longer place its framebuffers ("Failed to pin framebuffer with
error -12"), which has frozen a monitor until reboot. The default cap leaves
what the card's other users hold at startup, plus this process's own use
outside the cap; it reserves nothing for later growth and warns instead.

Settings come from the environment (gfx803-train.sh sets them from its
--vram-reserve / --vram-cap options):
  GFX803_VRAM_CAP=N|none   cap PyTorch at N MiB, or not at all; overrides
                           GFX803_VRAM_RESERVE
  GFX803_VRAM_RESERVE=auto|N
                           leave N MiB for everything else; auto [default]
                           leaves what other users hold at startup plus
                           RUNTIME_OVERHEAD_MIB
"""
import glob
import os
import runpy
import sys

# VRAM this process uses outside PyTorch's cap: the HIP context (about
# 140 MiB on an RX 590) and the code object of every PyTorch, MIOpen and shim
# kernel it loads. An SD1.5 LoRA run peaked 0.8-1.1 GB above PyTorch's
# reserved memory plus what the desktop held.
RUNTIME_OVERHEAD_MIB = 1024
# Below this, SD1.5 LoRA training at 512 px cannot run: its peak allocation
# is about 6.1 GB at batch 1 with gradient checkpointing, and even the
# smallest buckets and ranks need several GB.
MIN_CAP_MIB = 3072
# A desktop's VRAM use grows during a run (two monitors here: about 150 MiB
# locked, 600-700 MiB unlocked); with less spare than this at the expected
# peak, growth can push training buffers into system memory.
DISPLAY_HEADROOM_WARN_MIB = 512
WARN_OTHERS_MIB = 1536


def log(msg):
    print(f"[gfx803-vram] {msg}", file=sys.stderr, flush=True)


def amd_cards(drm_root):
    cards = []
    for card in sorted(glob.glob(os.path.join(drm_root, "card[0-9]*"))):
        if "-" in os.path.basename(card):
            continue
        try:
            with open(os.path.join(card, "device", "vendor")) as f:
                if f.read().strip() != "0x1002":
                    continue
        except OSError:
            continue
        cards.append(card)
    return cards


def connected_displays(drm_root, pci_bus_id=None):
    """Counts connected display connectors on the AMD card training uses.

    With several AMD cards and no way to tell which one PyTorch uses, every
    AMD card's connectors count, so the warnings err toward a desktop.
    """
    cards = amd_cards(drm_root)
    if pci_bus_id and len(cards) > 1:
        wanted = pci_bus_id.lower()
        matching = [c for c in cards
                    if os.path.basename(os.path.realpath(os.path.join(c, "device"))).lower() == wanted]
        cards = matching or cards
    n = 0
    for card in cards:
        name = os.path.basename(card)
        for status in glob.glob(os.path.join(drm_root, f"{name}-*", "status")):
            try:
                with open(status) as f:
                    n += f.read().strip() == "connected"
            except OSError:
                pass
    return n, [os.path.basename(c) for c in cards]


def card_vram_used_mib(drm_root, card):
    """VRAM in use on the card by every client, desktop included.

    hipMemGetInfo counts only compute (KFD) allocations, so it misses the
    compositor's framebuffers; amdgpu's own counter sees both.
    """
    try:
        with open(os.path.join(drm_root, card, "device", "mem_info_vram_used")) as f:
            return int(f.read()) >> 20
    except (OSError, ValueError):
        return None


def parse_mib(name, value):
    try:
        n = int(value)
    except ValueError:
        n = -1
    if n < 0:
        sys.exit(f"[gfx803-vram] {name} must be a whole number of MiB, got {value!r}")
    return n


def choose_cap(total_mib, others_mib, cap_setting, reserve_setting):
    """Returns (cap MiB or None, explanation, explicit) for the settings."""
    if cap_setting:
        if cap_setting.lower() == "none":
            return None, "no cap (GFX803_VRAM_CAP=none)", True
        cap = parse_mib("GFX803_VRAM_CAP", cap_setting)
        return min(cap, total_mib), f"fixed cap {cap} MiB (GFX803_VRAM_CAP)", True
    if reserve_setting and reserve_setting.lower() != "auto":
        reserve = parse_mib("GFX803_VRAM_RESERVE", reserve_setting)
        return max(total_mib - reserve, 0), f"reserve {reserve} MiB (GFX803_VRAM_RESERVE)", True
    reserve = others_mib + RUNTIME_OVERHEAD_MIB
    why = f"reserve {reserve} MiB (auto: {others_mib} in use + {RUNTIME_OVERHEAD_MIB} HIP runtime)"
    return max(total_mib - reserve, 0), why, False


def warnings(total_mib, others_mib, displays, cap_mib, explicit):
    """Returns (warning lines, fatal message or None) for a chosen cap."""
    warn = []
    if others_mib > WARN_OTHERS_MIB:
        warn.append(f"other processes already hold {others_mib} MiB (ComfyUI with a model "
                    "loaded, another training run?); training gets that much less")
    if cap_mib is None:
        if displays:
            warn.append(f"no cap with {displays} display(s) on this card: PyTorch's cache can "
                        "fill VRAM and push the desktop out, which slows training and can "
                        "freeze the display")
        return warn, None
    spare = total_mib - cap_mib - others_mib - RUNTIME_OVERHEAD_MIB
    if displays and spare < DISPLAY_HEADROOM_WARN_MIB:
        warn.append(f"this card drives {displays} display(s) and about {max(spare, 0)} MiB stays "
                    "free at training's expected peak; unlocking the screen or GPU-heavy apps "
                    "during the run can cause stutter or slow training. Training with the "
                    "desktop stopped avoids it")
    if cap_mib < MIN_CAP_MIB:
        msg = (f"the cap of {cap_mib} MiB is below the {MIN_CAP_MIB} MiB SD1.5 LoRA training "
               "needs at the least")
        if explicit:
            warn.append(msg + "; continuing because it was set explicitly")
        else:
            return warn, (msg + f" ({others_mib} MiB already in use on the card). Free VRAM "
                          "(close ComfyUI or other GPU programs), or set --vram-cap or "
                          "--vram-reserve to override")
    return warn, None


def main():
    script = sys.argv[1]
    cap_setting = os.environ.get("GFX803_VRAM_CAP", "").strip()
    reserve_setting = os.environ.get("GFX803_VRAM_RESERVE", "auto").strip()

    # Read before this process starts the HIP runtime, whose context would
    # otherwise count as another user of the card.
    used_before = {os.path.basename(c): card_vram_used_mib("/sys/class/drm", os.path.basename(c))
                   for c in amd_cards("/sys/class/drm")}

    import torch

    props = torch.cuda.get_device_properties(0)
    displays, cards = connected_displays("/sys/class/drm", getattr(props, "pci_bus_id", None))
    free, total = torch.cuda.mem_get_info()
    total_mib = total >> 20
    others_mib = used_before.get(cards[0]) if len(cards) == 1 else None
    if others_mib is None:
        others_mib = (total - free) >> 20

    cap_mib, why, explicit = choose_cap(total_mib, others_mib, cap_setting, reserve_setting)
    log(f"{'/'.join(cards) or 'no AMD DRM card found'}: {displays} connected display(s); "
        f"total {total_mib} MiB, in use by others {others_mib} MiB; {why} -> "
        f"{'PyTorch uncapped' if cap_mib is None else f'PyTorch cap {cap_mib} MiB'}")
    warn, fatal = warnings(total_mib, others_mib, displays, cap_mib, explicit)
    for w in warn:
        log("WARNING: " + w)
    if fatal:
        sys.exit("[gfx803-vram] ERROR: " + fatal)
    if cap_mib is not None:
        torch.cuda.set_per_process_memory_fraction(min(cap_mib / total_mib, 1.0))

    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    try:
        runpy.run_path(script, run_name="__main__")
    except torch.cuda.OutOfMemoryError:
        log("out of VRAM under the cap. Options: a smaller --resolution or --buckets range; "
            "freeing VRAM other programs hold; or a larger --vram-cap (or --vram-cap none), "
            "at the desktop's expense if this card drives one")
        raise


if __name__ == "__main__":
    main()
