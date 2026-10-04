#!/usr/bin/env python3
"""
Identify the GPU kernel(s) a compute queue is running (gfx8 / Polaris).

Reads KFD's hardware queue state (/sys/kernel/debug/kfd/hqds), finds the
active queues, and reads the AQL packets around each queue's read pointer.
The queue rings live in system RAM, which umr cannot read on this machine, so
the packets are read from the owning process's memory (/proc/<pid>/mem):
ROCm maps these buffers at the same address for the CPU and the GPU.

For each kernel-dispatch packet it records the launch shape, the kernel
arguments, and saves the kernel descriptor and the first 4 KiB of the kernel's
code as binary files (disassembled offline with llvm-mc -mcpu=gfx803).

Nothing on the GPU is halted or modified.

Used by gpu_hang_watch.sh at capture time, and with --state once a second to
see whether the busy queues still make progress. By hand, as root:
  sudo python3 gpu_hang_decode.py --selftest [--outdir DIR]   # while training
  python3 gpu_hang_decode.py --hqds FILE                      # queue summary only
  sudo python3 gpu_hang_decode.py --state                     # busy queues' pointers

Launch shapes worth knowing:
  workgroup 16x16x1              -> the SGEMM shim's tile kernel
                                    (grid z > 1 = the batched/attention variant)
  workgroup 64/128/256/512 x1x1  -> typically PyTorch / MIOpen kernels
"""
import argparse
import glob
import os
import re
import struct
import subprocess
import sys

# gfx8 CP_HQD_* register byte offsets as printed by debugfs kfd/hqds
R_ACTIVE, R_VMID = 0xC91C, 0xC920
R_PQ_BASE, R_PQ_BASE_HI, R_PQ_RPTR = 0xC934, 0xC938, 0xC93C
R_PQ_WPTR, R_PQ_CONTROL = 0xC954, 0xC958
R_EOP_RPTR, R_EOP_WPTR_MEM = 0xC9B4, 0xC9E4

HISTORY = tuple(range(-12, 1))   # packets before the read pointer, and the next one

PACKET_TYPES = {0: "vendor", 1: "invalid", 2: "KERNEL_DISPATCH", 3: "BARRIER_AND",
                4: "agent_dispatch", 5: "BARRIER_OR"}


def parse_hqds(text):
    queues = []
    for block in re.split(r"\n(?=\s*Inst \d+,|\s*SDMA)", text):
        m = re.search(r"CP Pipe (\d+), Queue (\d+)", block)
        if not m:
            continue
        regs = {}
        for line in block.splitlines():
            mm = re.match(r"\s*([0-9a-fA-F]{8}):((?:\s+[0-9a-fA-F]{8})+)", line)
            if mm:
                base = int(mm.group(1), 16)
                for i, v in enumerate(mm.group(2).split()):
                    regs[base + 4 * i] = int(v, 16)
        if regs.get(R_ACTIVE) != 1:
            continue
        ctl = regs.get(R_PQ_CONTROL, 0)
        ring_dw = 1 << ((ctl & 0x3F) + 1)
        rptr, wptr = regs.get(R_PQ_RPTR, 0), regs.get(R_PQ_WPTR, 0)
        queues.append({
            "name": f"Pipe {m.group(1)} Queue {m.group(2)}",
            "vmid": regs.get(R_VMID, 0) & 0xF,
            "ring": ((regs.get(R_PQ_BASE_HI, 0) << 32) | regs.get(R_PQ_BASE, 0)) << 8,
            "ring_dw": ring_dw,
            "rptr": rptr % ring_dw,
            "wptr": wptr % ring_dw,
            "pending": ((wptr - rptr) % ring_dw) // 16,
            "eop_rptr": regs.get(R_EOP_RPTR, 0) & 0xFFFF,
            "eop_wptr": regs.get(R_EOP_WPTR_MEM, 0) & 0xFFFF,
        })
    return queues


def describe(q):
    inflight = q["eop_rptr"] != q["eop_wptr"]
    return (f"{q['name']}: vmid {q['vmid']}, ring 0x{q['ring']:x} ({q['ring_dw'] * 4 // 1024} KiB), "
            f"rptr 0x{q['rptr']:x}, wptr 0x{q['wptr']:x}, {q['pending']} packets queued, "
            f"end-of-pipe {q['eop_rptr']:#x}/{q['eop_wptr']:#x}"
            + (" (work in flight)" if inflight or q["pending"] else " (idle)"))


def kfd_pids():
    pids = []
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd) == "/dev/kfd":
                pids.append(int(fd.split("/")[2]))
        except OSError:
            pass
    return sorted(set(pids))


LAST_ERR = {}


def mem_read(pid, addr, size):
    try:
        with open(f"/proc/{pid}/mem", "rb", buffering=0) as f:
            f.seek(addr)
            data = f.read(size)
            if len(data) != size:
                LAST_ERR[pid] = f"short read ({len(data)} of {size} bytes)"
                return None
            return data
    except OSError as e:
        LAST_ERR[pid] = f"{type(e).__name__}: {e}"
        return None


def mapping_of(pid, addr):
    """The /proc/<pid>/maps line containing addr, or None."""
    try:
        with open(f"/proc/{pid}/maps") as f:
            for line in f:
                lo, hi = (int(x, 16) for x in line.split()[0].split("-"))
                if lo <= addr < hi:
                    return line.rstrip()
    except OSError as e:
        return f"<maps unreadable: {e}>"
    return None


def gpu_mappings(pid, size=None):
    """Mappings of GPU device files (render node / kfd), optionally of one size."""
    res = []
    try:
        with open(f"/proc/{pid}/maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6 and parts[5].startswith(("/dev/dri", "/dev/kfd")):
                    lo, hi = (int(x, 16) for x in parts[0].split("-"))
                    if size is None or hi - lo == size:
                        res.append((lo, hi, line.rstrip()))
    except OSError:
        pass
    return res


def plausible_ring(data):
    pkts = [struct.unpack("<16I", data[i:i + 64]) for i in range(0, len(data), 64)]
    return bool(pkts) and all((w[0] & 0xFF) in PACKET_TYPES for w in pkts) and \
        any((w[0] & 0xFF) in (2, 3, 5) or looks_like_dispatch(w) for w in pkts)


def looks_like_dispatch(w):
    """A kernel-dispatch body: sane workgroup/grid sizes and code/kernarg pointers.
    The packet processor overwrites the header of consumed packets with
    INVALID (type 1), so the body is all that identifies them."""
    wx, wy, wz = w[1] & 0xFFFF, w[1] >> 16, w[2] & 0xFFFF
    return (1 <= wx <= 1024 and 1 <= wy <= 1024 and 1 <= wz <= 1024 and wx * wy * wz <= 1024
            and w[3] >= 1 and w[4] >= 1 and w[5] >= 1
            and (w[8] | w[9]) and (w[10] | w[11]))


def decode_packet(pkt):
    w = struct.unpack("<16I", pkt)
    ptype = w[0] & 0xFF
    info = {"type": PACKET_TYPES.get(ptype, f"type{ptype}")}
    if ptype == 1 and looks_like_dispatch(w):
        ptype = 2
        info["type"] = "KERNEL_DISPATCH (consumed)"
    if ptype == 2:
        info.update(
            workgroup=(w[1] & 0xFFFF, w[1] >> 16, w[2] & 0xFFFF),
            grid=(w[3], w[4], w[5]),
            lds=w[7], scratch=w[6],
            kernel_object=w[8] | (w[9] << 32),
            kernarg=w[10] | (w[11] << 32),
            signal=w[14] | (w[15] << 32),
        )
        if info["workgroup"] == (16, 16, 1) and w[7] == 2048:
            info["likely"] = ("SGEMM shim tile kernel, batched (attention)" if w[5] > 1
                              else "SGEMM shim tile kernel") + \
                f", C is about {w[4]}x{w[3]} (rows x cols, rounded up to 16)"
    elif ptype in (3, 5):
        info["dep_signals"] = [hex(w[4 + 2 * i] | (w[5 + 2 * i] << 32)) for i in range(5)]
        info["signal"] = w[14] | (w[15] << 32)
    return info


def fmt(info):
    return {k: (hex(v) if isinstance(v, int) and k in ("kernel_object", "kernarg", "signal") else v)
            for k, v in info.items()}


def inspect_queue(q, pids, out, outdir, qi):
    ring_bytes = q["ring_dw"] * 4
    rp = q["rptr"] * 4
    offsets = []
    pid, packets, readable = None, None, []
    # gfx8: ROCm maps a ring of N bytes twice back to back for the GPU and
    # reports 2N to the hardware, while the CPU maps it only once. Try the
    # reported size first, then half of it.
    for size in (ring_bytes, ring_bytes // 2):
        # the last four packets the CP fetched (the newest may still be
        # running) and the next one waiting
        offs = [(rp + d * 64) % size for d in HISTORY]
        for p in pids:
            maps = [mapping_of(p, q["ring"] + o) for o in offs]
            if any(m is None or m.split()[1].startswith("---") for m in maps):
                continue   # not mapped, or only an inaccessible reservation
            data = [mem_read(p, q["ring"] + o, 64) for o in offs]
            if all(data) and any(any(d) for d in data):
                readable.append((p, data, offs, size))
                if plausible_ring(b"".join(data)):
                    pid, packets, offsets = p, data, offs
                    out.append(f"  ring size used for wrap-around: {size // 1024} KiB\n")
                    break
        if pid is not None:
            break
    if pid is None and readable:
        pid, packets, offsets, size = readable[0]
        readable = [(pid, packets)]
    if readable and packets is not None and not plausible_ring(b"".join(packets)):
        out.append("  NOTE: packets at the read pointer do not look like a normal AQL "
                   "queue; raw dwords follow each packet\n")
    if pid is None:
        offsets = [(rp + d * 64) % ring_bytes for d in HISTORY]
        out.append(f"  could not read the ring at 0x{q['ring']:x} directly; details per process:\n")
        for p in pids:
            out.append(f"    pid {p}: mapping {mapping_of(p, q['ring'])!r}, last error {LAST_ERR.get(p)!r}\n")
        # Fallback: the CPU may map the ring elsewhere. Try every GPU mapping
        # of exactly the ring's size.
        for p in pids:
            for lo, hi, line in gpu_mappings(p, ring_bytes):
                data = [mem_read(p, lo + o, 64) for o in offsets]
                ok = all(data) and plausible_ring(b"".join(data))
                if ok:
                    out.append(f"    candidate pid {p} {line!r}: MATCH\n")
                if ok and pid is None:
                    pid, packets = p, data
        if pid is None:
            for p in pids:
                maps = gpu_mappings(p)
                out.append(f"    pid {p}: {len(maps)} GPU mappings; first few:\n")
                out.extend(f"      {m[2]}\n" for m in maps[:8])
            return
        out.append("  (ring found at a different CPU address; kernel code/args may not be readable)\n")
    out.append(f"  ring read from pid {pid}\n")
    seen = {}
    for o, d, pkt in zip(offsets, HISTORY, packets):
        tag = "at read pointer" if d == 0 else f"fetched {-d} back"
        info = decode_packet(pkt)
        out.append(f"  packet @ring+0x{o:x} [{tag}]: {fmt(info)}\n")
        out.append("    raw: " + " ".join(f"{x:08x}" for x in struct.unpack("<16I", pkt)) + "\n")
        if not info["type"].startswith("KERNEL_DISPATCH"):
            continue
        args = mem_read(pid, info["kernarg"], 128)
        if args:
            words = struct.unpack("<32I", args)
            out.append("    kernargs: " + " ".join(f"{x:08x}" for x in words) + "\n")
        ko = info["kernel_object"]
        if ko in seen:
            out.append(f"    kernel: same as {seen[ko]}\n")
            continue
        name = f"q{qi}_k{len(seen)}"
        seen[ko] = name
        desc = mem_read(pid, ko, 64)
        if not desc:
            out.append(f"    kernel descriptor at {ko:#x}: not readable from pid {pid}\n")
            continue
        entry = ko + struct.unpack_from("<q", desc, 16)[0]
        code = mem_read(pid, entry, 4096)
        out.append(f"    kernel {name}: descriptor {ko:#x}, code {entry:#x}, "
                   f"code bytes {'saved' if code else 'not readable'}\n")
        if outdir:
            with open(os.path.join(outdir, f"{name}_descriptor.bin"), "wb") as f:
                f.write(desc)
            if code:
                with open(os.path.join(outdir, f"{name}_code.bin"), "wb") as f:
                    f.write(code)
        if not code:
            try:
                r = subprocess.run(["umr", "-vdis", f"{q['vmid']}@0x{entry:x}", "1024"],
                                   capture_output=True, text=True, timeout=60)
                out.append(r.stdout + r.stderr)
            except Exception as e:  # noqa: BLE001
                out.append(f"    umr disassembly failed: {e}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hqds", help="read queue state from this file instead of debugfs")
    ap.add_argument("--selftest", action="store_true", help="decode active queues even if not busy")
    ap.add_argument("--outdir", help="save kernel descriptors/code here")
    ap.add_argument("--state", action="store_true",
                    help="print one line per busy queue: name, read pointer, end-of-pipe read pointer")
    args = ap.parse_args()
    text = open(args.hqds or "/sys/kernel/debug/kfd/hqds").read()
    queues = parse_hqds(text)
    if args.state:
        # The write pointer is left out on purpose: the host keeps queueing work
        # behind a stuck dispatch, so only the read side shows whether the
        # GPU is still consuming packets.
        for q in queues:
            if q["pending"] or q["eop_rptr"] != q["eop_wptr"]:
                print(f"{q['name'].replace(' ', '')} rptr=0x{q['rptr']:x} eop=0x{q['eop_rptr']:x}")
        return
    out = [describe(q) + "\n" for q in queues] or ["No active compute queues.\n"]
    if not args.hqds:
        pids = kfd_pids()
        out.append(f"processes with /dev/kfd open: {pids}\n")
        targets = [q for q in queues if q["pending"] or q["eop_rptr"] != q["eop_wptr"]]
        if args.selftest and not targets:
            targets = queues
        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
        for qi, q in enumerate(targets):
            out.append(f"\n=== {q['name']} ===\n")
            inspect_queue(q, pids, out, args.outdir, qi)
    sys.stdout.write("".join(out))


if __name__ == "__main__":
    main()
