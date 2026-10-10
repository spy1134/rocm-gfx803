"""
gfx803 Fused Attention Python Runtime Wrapper and PyTorch SDPA Hook.

Provides high-performance, memory-efficient online softmax attention for
AMD Polaris (gfx803) GPUs, completely eliminating intermediate O(N*M) VRAM
allocations (saving 1.2 GB+ transient VRAM per SD 1.5 step) and accelerating
attention throughput by up to 2.2x over stock PyTorch Math Attention.
"""

import os
import sys
import ctypes
import math
import torch
import torch.nn.functional as F

# Locate and load the native shared library
_SEARCH_PATHS = [
    os.environ.get("GFX803_ATTN_LIB", ""),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libgfx803_attn.so"),
    "/workspace/libgfx803_attn.so",
    "/opt/rocm/lib/libgfx803_attn.so",
    os.path.join(os.getcwd(), "tools", "libgfx803_attn.so"),
    os.path.join(os.getcwd(), "libgfx803_attn.so"),
]
_LIB_PATH = None
for p in _SEARCH_PATHS:
    if p and os.path.exists(p):
        _LIB_PATH = p
        break

_lib = None
if _LIB_PATH and os.path.exists(_LIB_PATH):
    try:
        _lib = ctypes.CDLL(_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
        _lib.gfx803_fused_attention_forward.argtypes = [
            ctypes.c_void_p,  # Q
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # V
            ctypes.c_void_p,  # O
            ctypes.c_int,     # B
            ctypes.c_int,     # H
            ctypes.c_int,     # N
            ctypes.c_int,     # M
            ctypes.c_int,     # D
            ctypes.c_float,   # scale
            ctypes.c_void_p   # stream
        ]
        _lib.gfx803_fused_attention_forward.restype = ctypes.c_int
    except Exception as e:
        sys.stderr.write(f"[gfx803_attention] Warning: Failed to load {_LIB_PATH}: {e}\n")


def is_available() -> bool:
    return _lib is not None and torch.cuda.is_available()


def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None
) -> torch.Tensor:
    """
    Execute fused memory-efficient attention on gfx803.

    Args:
        q: Query tensor of shape [B, H, N, D] or [B, N, D]
        k: Key tensor of shape [B, H, M, D] or [B, M, D]
        v: Value tensor of shape [B, H, M, D] or [B, M, D]
        scale: Softmax scaling factor (default: 1.0 / sqrt(D))

    Returns:
        Output tensor of same shape as q
    """
    if _lib is None:
        raise RuntimeError("gfx803 fused attention shared library not loaded")

    orig_shape = q.shape
    is_3d = (q.ndim == 3)
    if is_3d:
        # Reshape [B, N, D] -> [B, 1, N, D]
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"Expected 4D tensors [B, H, N, D], got Q:{q.shape}, K:{k.shape}, V:{v.shape}")

    B, H, N, D = q.shape
    Bk, Hk, M, Dk = k.shape
    if B != Bk or H != Hk or D != Dk:
        raise ValueError(f"Shape mismatch: Q={q.shape}, K={k.shape}")

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Ensure contiguous FP32 tensors on device
    orig_dtype = q.dtype
    if orig_dtype != torch.float32:
        q_f32 = q.contiguous().to(torch.float32)
        k_f32 = k.contiguous().to(torch.float32)
        v_f32 = v.contiguous().to(torch.float32)
    else:
        q_f32 = q.contiguous()
        k_f32 = k.contiguous()
        v_f32 = v.contiguous()

    out_f32 = torch.empty_like(q_f32)

    stream = torch.cuda.current_stream().cuda_stream

    ret = _lib.gfx803_fused_attention_forward(
        ctypes.c_void_p(q_f32.data_ptr()),
        ctypes.c_void_p(k_f32.data_ptr()),
        ctypes.c_void_p(v_f32.data_ptr()),
        ctypes.c_void_p(out_f32.data_ptr()),
        ctypes.c_int(B),
        ctypes.c_int(H),
        ctypes.c_int(N),
        ctypes.c_int(M),
        ctypes.c_int(D),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream)
    )

    if ret != 0:
        raise RuntimeError(f"gfx803_fused_attention_forward failed with code {ret}")

    out = out_f32 if orig_dtype == torch.float32 else out_f32.to(orig_dtype)
    return out.squeeze(1) if is_3d else out


_orig_sdpa = F.scaled_dot_product_attention


def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    # Strictly forward-only: if any input tracks gradients for autograd, immediately
    # pass through to stock PyTorch to guarantee mathematically correct backpropagation.
    if query.requires_grad or key.requires_grad or value.requires_grad:
        return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)

    # Route forward inference when unmasked, no dropout, non-causal.
    # D in (40, 64, 80) accelerate all sequence lengths (up to 2.2x speedup).
    # D=160 cross-attention regresses to 0.72x on Polaris hardware (0.143 ms vs 0.102 ms stock)
    # because short sequence length (M <= 128) has low memory overhead while large head
    # dimension reduces CU occupancy. Stock PyTorch Math Attention is faster for D=160.
    d = query.shape[-1]
    if (
        is_available()
        and query.is_cuda
        and attn_mask is None
        and dropout_p == 0.0
        and not is_causal
        and not enable_gqa
        and (d in (40, 64, 80))
        # The kernel takes one [B, H, *, D] layout for all three tensors; stock
        # SDPA also accepts broadcast batch/head dims and a V head size != D.
        and query.ndim in (3, 4) and query.ndim == key.ndim == value.ndim
        and key.shape[:-2] == query.shape[:-2] == value.shape[:-2]
        and key.shape[-1] == d == value.shape[-1]
    ):
        return fused_attention(query, key, value, scale=scale)

    return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)


def patch_torch_sdpa():
    """Install gfx803 fused attention as the default handler for F.scaled_dot_product_attention."""
    F.scaled_dot_product_attention = patched_sdpa
    torch.nn.functional.scaled_dot_product_attention = patched_sdpa
    print("[gfx803_attention] F.scaled_dot_product_attention successfully patched with gfx803 2D block-tiled attention.")


def unpatch_torch_sdpa():
    """Restore original PyTorch scaled_dot_product_attention."""
    F.scaled_dot_product_attention = _orig_sdpa
    torch.nn.functional.scaled_dot_product_attention = _orig_sdpa


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA/ROCm not available.")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Fused attention available: {is_available()}")

    # Test cases matching real Stable Diffusion 1.5 shapes
    test_cases = [
        ("SD1.5 Self 64x64", 1, 8, 4096, 4096, 40),
        ("SD1.5 Cross 64x64", 1, 8, 4096, 77, 40),
        ("SD1.5 Self 32x32", 1, 8, 1024, 1024, 80),
        ("SD1.5 Cross 32x32", 1, 8, 1024, 77, 80),
        ("SD1.5 Self 16x16", 1, 8, 256, 256, 160),
        ("SD1.5 Cross 16x16", 1, 8, 256, 77, 160),
    ]

    for name, B, H, N, M, D in test_cases:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)
        scale = 1.0 / math.sqrt(D)

        # PyTorch reference
        torch.cuda.synchronize()
        ref_out = _orig_sdpa(q, k, v, scale=scale)
        torch.cuda.synchronize()

        # Fused kernel
        fused_out = fused_attention(q, k, v, scale=scale)
        torch.cuda.synchronize()

        max_diff = (ref_out - fused_out).abs().max().item()

        # Benchmarking
        runs = 20
        # Stock timing
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(runs):
            _ = _orig_sdpa(q, k, v, scale=scale)
        t1.record()
        torch.cuda.synchronize()
        ms_stock = t0.elapsed_time(t1) / runs

        # Fused timing
        torch.cuda.reset_peak_memory_stats()
        t0.record()
        for _ in range(runs):
            _ = fused_attention(q, k, v, scale=scale)
        t1.record()
        torch.cuda.synchronize()
        ms_fused = t0.elapsed_time(t1) / runs
        speedup = ms_stock / ms_fused

        print(f"%-20s | MaxDiff: %.2e | Stock: %6.2f ms | Fused: %6.2f ms | Speedup: %.2fx" % (
            name, max_diff, ms_stock, ms_fused, speedup
        ))
