#!/usr/bin/env python3
"""
GPU-only stress loop for the batched GEMM path (torch.bmm / baddbmm), which the
SGEMM shim's strided-batched kernel handles. Runs sd-scripts' "normal" attention at
SD1.5's shapes (self- and cross-attention at every UNet level), forward and
backward, repeatedly, with no CPU reference - the goal is to reproduce the GPU
hang seen in training quickly, not to check numbers (gfx803_unet_check.py does
that).

Run inside the container, with ComfyUI idle, and on the host the tools in
tools/gpu-hang-capture/ (gpu_monitor.sh, and sudo bash gpu_hang_watch.sh to
capture the stuck queue if it hangs):

  HSA_DISABLE_FRAGMENT_ALLOCATOR=0 \
    python -u /opt/gfx803-training/gfx803_attn_stress.py --minutes 30

Progress is printed every --report iterations; the last line printed before a
hang shows how far it got.
"""
import argparse
import os
import time

import torch

# (name, batch*heads, query_len, key_len, head_dim) at 512x512 (64x64 latent).
# Cross-attention keys use 227 tokens (sd-scripts --max_token_length=225).
SHAPES = [
    ("self  64x64", 8, 4096, 4096, 40),
    ("cross 64x64", 8, 4096, 227, 40),
    ("self  32x32", 8, 1024, 1024, 80),
    ("cross 32x32", 8, 1024, 227, 80),
    ("self  16x16", 8, 256, 256, 160),
    ("cross 16x16", 8, 256, 227, 160),
    ("self   8x8 ", 8, 64, 64, 160),
    ("cross  8x8 ", 8, 64, 227, 160),
]


def attention(q, k, v, scale):
    # sd-scripts original_unet CrossAttention (normal path): baddbmm -> softmax -> bmm
    scores = torch.baddbmm(
        torch.empty(q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device),
        q, k.transpose(1, 2), beta=0, alpha=scale)
    return torch.bmm(scores.softmax(dim=-1), v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--report", type=int, default=25, help="print progress every N iterations")
    ap.add_argument("--no-sync", action="store_true",
                    help="don't synchronize after each iteration (closer to training's timing)")
    args = ap.parse_args()

    print(f"torch {torch.__version__}, HIP {torch.version.hip}, device {torch.cuda.get_device_name(0)}")
    for var in ("LD_PRELOAD", "GFX803_SGEMM_SHIM_DISABLE", "GFX803_SGEMM_SHIM_NO_BATCHED",
                "HSA_DISABLE_FRAGMENT_ALLOCATOR"):
        print(f"  {var}={os.environ.get(var, '<unset>')}")

    dev = "cuda"
    tensors = []
    for name, bh, sq, sk, d in SHAPES:
        q = torch.randn(bh, sq, d, device=dev, requires_grad=True)
        k = torch.randn(bh, sk, d, device=dev, requires_grad=True)
        v = torch.randn(bh, sk, d, device=dev, requires_grad=True)
        go = torch.randn(bh, sq, d, device=dev)
        tensors.append((name, q, k, v, go, d ** -0.5))

    deadline = time.time() + args.minutes * 60
    it = 0
    t0 = time.time()
    while time.time() < deadline:
        for name, q, k, v, go, scale in tensors:
            q.grad = k.grad = v.grad = None
            out = attention(q, k, v, scale)
            (out * go).sum().backward()
        if not args.no_sync:
            torch.cuda.synchronize()
        it += 1
        if it % args.report == 0:
            torch.cuda.synchronize()
            finite = all(torch.isfinite(t.grad).all().item() for _, q, k, v, _, _ in tensors for t in (q, k, v))
            print(f"iter {it:6d}  {time.time() - t0:7.1f}s  grads finite: {finite}", flush=True)
    torch.cuda.synchronize()
    print(f"Completed {it} iterations in {time.time() - t0:.1f}s without a hang.")


if __name__ == "__main__":
    main()
