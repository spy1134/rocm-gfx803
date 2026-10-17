#!/usr/bin/env python3
"""The ComfyUI node's fused attention (tools/libgfx803_attn.so) against float64 on the CPU.

Run on the GPU under test:  python3 -u attention_check.py [--lib PATH] [--quick]
Exit status 0 only if every case is within tolerance.

Realistic inputs (SD 1.5 and SDXL shapes, odd sizes, tails, fp16-rounded inputs) must
land within 1e-4 of the reference, normalised by its largest magnitude. The large-logit
cases (inputs scaled 50x or 200x, so the softmax is nearly one-hot) are ill-conditioned:
one rounding step in a score of magnitude ~1000 moves the output visibly, and an exact
fp32 computation on the CPU already lands 2e-4 to 5e-4 away. Those cases are judged
against that CPU fp32 error and fail above 4x it.
"""
import argparse
import ctypes
import math
import os
import sys

import torch

TOL = 1e-4
STRESS_FACTOR = 4.0

f32, f16 = torch.float32, torch.float16
# (label, B, H, N, M, D, dtype, input scale)
CASES = [
    ("SD1.5 self 64x64 CFG",        2, 8, 4096, 4096, 40, f16, 1.0),
    ("SD1.5 self 64x64",            1, 8, 4096, 4096, 40, f32, 1.0),
    ("SD1.5 cross 64x64",           1, 8, 4096,   77, 40, f32, 1.0),
    ("SD1.5 self 32x32",            1, 8, 1024, 1024, 80, f32, 1.0),
    ("SD1.5 cross 32x32",           1, 8, 1024,   77, 80, f16, 1.0),
    ("SD1.5 self 16x16 (D 160)",    1, 8,  256,  256, 160, f32, 1.0),
    ("SDXL self 4096",              1, 10, 4096, 4096, 64, f16, 1.0),
    ("SDXL self 1024",              1, 20, 1024, 1024, 64, f32, 1.0),
    ("SDXL cross 4096",             1, 10, 4096,   77, 64, f32, 1.0),
    ("odd 77x77 D40",               1, 2,    77,   77, 40, f32, 1.0),
    ("odd 1000x1000 D40",           1, 2,  1000, 1000, 40, f32, 1.0),
    ("odd 4097x4097 D40",           1, 1,  4097, 4097, 40, f32, 1.0),
    ("odd 1000x1000 D64",           1, 2,  1000, 1000, 64, f32, 1.0),
    ("odd 4097x4097 D64",           1, 1,  4097, 4097, 64, f32, 1.0),
    ("odd 1000x1000 D80",           1, 2,  1000, 1000, 80, f32, 1.0),
    ("tail 129x33 D40",             1, 3,   129,   33, 40, f32, 1.0),
    ("tail 1x1 D40",                1, 1,     1,    1, 40, f32, 1.0),
    ("tail 1x1 D64",                1, 1,     1,    1, 64, f32, 1.0),
    ("tail 5x100 D80",              1, 1,     5,  100, 80, f32, 1.0),
    ("large logits D40 x50",        1, 2,  1024, 1024, 40, f32, 50.0),
    ("large logits D64 x50",        1, 2,  1024, 1024, 64, f32, 50.0),
    ("large logits D80 x50",        1, 2,  1024, 1024, 80, f32, 50.0),
    ("large logits D40 M77 x200",   1, 2,  1024,   77, 40, f32, 200.0),
]


def default_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "libgfx803_attn.so"))


def reference(q, k, v, scale, dtype=torch.float64):
    q, k, v = (t.detach().cpu().to(dtype) for t in (q, k, v))
    out = torch.empty_like(q)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            out[b, h] = torch.softmax((q[b, h] @ k[b, h].T) * scale, dim=-1) @ v[b, h]
    return out


def norm_err(out, ref):
    out = out.detach().cpu().double()
    if not torch.isfinite(out).all():
        return math.inf
    return (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=default_lib())
    ap.add_argument("--quick", action="store_true", help="skip the 4096-token cases")
    ap.add_argument("--vram-frac", type=float, default=0.4)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no GPU visible"); sys.exit(2)
    torch.cuda.set_per_process_memory_fraction(args.vram_frac)
    fn = ctypes.CDLL(args.lib).gfx803_fused_attention_forward
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    print(f"device: {torch.cuda.get_device_name(0)}  lib {args.lib}", flush=True)

    failed = 0
    torch.manual_seed(1234)
    for label, B, H, N, M, D, dtype, mag in CASES:
        if args.quick and max(N, M) >= 4096:
            continue
        scale = 1.0 / math.sqrt(D)
        q = (torch.randn(B, H, N, D, device="cuda") * mag).to(dtype).float().contiguous()
        k = (torch.randn(B, H, M, D, device="cuda") * mag).to(dtype).float().contiguous()
        v = torch.randn(B, H, M, D, device="cuda").to(dtype).float().contiguous()
        out = torch.empty_like(q)
        rc = fn(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), B, H, N, M, D,
                ctypes.c_float(scale), torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        ref = reference(q, k, v, scale)
        err = norm_err(out, ref) if rc == 0 else math.inf
        if mag > 1.0:
            limit = STRESS_FACTOR * norm_err(reference(q, k, v, scale, torch.float32), ref)
        else:
            limit = TOL
        ok = err <= limit
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label:<28} B{B} H{H} N{N} M{M} D{D} "
              f"{str(dtype).split('.')[-1]:<7} rc {rc} err {err:.2e} limit {limit:.2e}", flush=True)
        del q, k, v, out
        torch.cuda.empty_cache()
    print(f"RESULT: {'FAIL' if failed else 'PASS'} ({failed} failing)", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
