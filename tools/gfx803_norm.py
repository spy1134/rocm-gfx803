# tools/gfx803_norm.py
# Python bindings for gfx803 fused GroupNorm + SiLU kernel.

import os
import ctypes
import torch
import torch.nn.functional as F

_SO_DIR = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_SO_DIR, "libgfx803_norm.so")

_LIB = None
if os.path.exists(_SO_PATH):
    try:
        _LIB = ctypes.CDLL(_SO_PATH)
        _LIB.gfx803_group_norm_fwd.argtypes = [
            ctypes.c_void_p, # input
            ctypes.c_void_p, # gamma
            ctypes.c_void_p, # beta
            ctypes.c_void_p, # output
            ctypes.c_int,    # N
            ctypes.c_int,    # C
            ctypes.c_int,    # H
            ctypes.c_int,    # W
            ctypes.c_int,    # num_groups
            ctypes.c_float,  # eps
            ctypes.c_bool,   # apply_silu
            ctypes.c_void_p, # stream
        ]
        _LIB.gfx803_group_norm_fwd.restype = None
    except Exception as e:
        _LIB = None

_orig_f_group_norm = F.group_norm

def gfx803_group_norm(input: torch.Tensor, num_groups: int, weight: torch.Tensor = None,
                      bias: torch.Tensor = None, eps: float = 1e-5, apply_silu: bool = False) -> torch.Tensor:
    # Fall back to PyTorch if not on CUDA/HIP, not float32, or backward pass required
    if (_LIB is None or not input.is_cuda or input.dtype != torch.float32 or 
        input.requires_grad or (weight is not None and weight.requires_grad) or
        (bias is not None and bias.requires_grad)):
        out = _orig_f_group_norm(input, num_groups, weight, bias, eps)
        if apply_silu:
            out = F.silu(out)
        return out

    # Flatten spatial dims to H, W if input has more/fewer dimensions
    orig_shape = input.shape
    if input.dim() == 4:
        N, C, H, W = input.shape
    elif input.dim() == 3:
        N, C, L = input.shape
        H, W = 1, L
    elif input.dim() == 2:
        N, C = input.shape
        H, W = 1, 1
    else:
        out = _orig_f_group_norm(input, num_groups, weight, bias, eps)
        return F.silu(out) if apply_silu else out

    # Guard: unsupported group partition or LDS capacity exceeded (>128 channels/group)
    if C % num_groups != 0 or (C // num_groups) > 128:
        out = _orig_f_group_norm(input, num_groups, weight, bias, eps)
        return F.silu(out) if apply_silu else out

    # Ensure contiguous
    x_c = input.contiguous()
    w_ptr = weight.contiguous().data_ptr() if weight is not None else 0
    b_ptr = bias.contiguous().data_ptr() if bias is not None else 0
    
    out = torch.empty_like(x_c)
    stream = torch.cuda.current_stream().cuda_stream

    _LIB.gfx803_group_norm_fwd(
        ctypes.c_void_p(x_c.data_ptr()),
        ctypes.c_void_p(w_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(N),
        ctypes.c_int(C),
        ctypes.c_int(H),
        ctypes.c_int(W),
        ctypes.c_int(num_groups),
        ctypes.c_float(eps),
        ctypes.c_bool(apply_silu),
        ctypes.c_void_p(stream),
    )
    
    return out.reshape(orig_shape)

def gfx803_group_norm_silu(input: torch.Tensor, num_groups: int, weight: torch.Tensor = None,
                           bias: torch.Tensor = None, eps: float = 1e-5) -> torch.Tensor:
    return gfx803_group_norm(input, num_groups, weight, bias, eps, apply_silu=True)
