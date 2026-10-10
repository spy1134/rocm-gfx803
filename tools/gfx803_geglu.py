# tools/gfx803_geglu.py
# Python wrapper for gfx803 fused GEGLU kernel.

import os
import ctypes
import torch
import torch.nn.functional as F

_SO_DIR = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_SO_DIR, "libgfx803_geglu.so")

_LIB = None
if os.path.exists(_SO_PATH):
    try:
        _LIB = ctypes.CDLL(_SO_PATH)
        _LIB.gfx803_geglu_fwd.argtypes = [
            ctypes.c_void_p, # input
            ctypes.c_void_p, # output
            ctypes.c_int,    # total_rows
            ctypes.c_int,    # D
            ctypes.c_bool,   # approximate_tanh
            ctypes.c_void_p, # stream
        ]
        _LIB.gfx803_geglu_fwd.restype = None
    except Exception as e:
        _LIB = None

def gfx803_geglu(x: torch.Tensor, approximate_tanh: bool = False) -> torch.Tensor:
    # x has shape [..., 2*D]
    dim = x.shape[-1]
    if (dim % 2 != 0) or (_LIB is None) or (not x.is_cuda) or (x.dtype != torch.float32) or x.requires_grad:
        # Stock PyTorch fallback (matches ComfyUI exact erf when approximate_tanh=False)
        x1, x2 = x.chunk(2, dim=-1)
        if approximate_tanh:
            return x1 * F.gelu(x2, approximate='tanh')
        else:
            return x1 * F.gelu(x2)
    
    D = dim // 2
    if D % 4 != 0:
        x1, x2 = x.chunk(2, dim=-1)
        if approximate_tanh:
            return x1 * F.gelu(x2, approximate='tanh')
        else:
            return x1 * F.gelu(x2)
        
    orig_shape = list(x.shape)
    orig_shape[-1] = D
    
    total_rows = x.numel() // (2 * D)
    x_c = x.contiguous()
    out = torch.empty(orig_shape, device=x.device, dtype=torch.float32)
    
    stream = torch.cuda.current_stream().cuda_stream
    _LIB.gfx803_geglu_fwd(
        ctypes.c_void_p(x_c.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(total_rows),
        ctypes.c_int(D),
        ctypes.c_bool(approximate_tanh),
        ctypes.c_void_p(stream),
    )
    
    return out
