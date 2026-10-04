# gfx803_layernorm_fix.py
# WORKAROUND, not a fix: PyTorch's own HIP layer_norm kernel computes wrong
# results on gfx803. Called directly (torch.ops.aten.native_layer_norm, fp32),
# it disagrees with a float64 CPU reference in every one of 80 runs across
# SD1.5's shapes ((2, 4096, 320), (2, 77, 768) and others): forward errors up to
# 8.5 on unit-scale outputs, backward up to 51% relative. Every transformer
# block in SD1.5 uses it, so without this both image generation and training
# compute garbage. GroupNorm with one group over the normalized dimensions is
# mathematically the same operation and computes correctly here, so GPU
# layer_norm calls are routed through it. To reproduce, compare
# native_layer_norm on the GPU with the CPU on random fp32 input. The source fix
# belongs in the PyTorch build.
import torch
import torch.nn.functional as F

_orig_torch_layer_norm = torch.layer_norm

def safe_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5, cudnn_enable=True):
    if not input.is_cuda:
        return _orig_torch_layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable)

    C = 1
    for d in normalized_shape:
        C *= d
    orig_shape = input.shape
    x = input.reshape(-1, C, 1, 1)
    w = weight.reshape(C) if weight is not None else None
    b = bias.reshape(C) if bias is not None else None
    out = F.group_norm(x, 1, w, b, eps)
    return out.reshape(orig_shape)

F.layer_norm = safe_layer_norm
torch.nn.functional.layer_norm = safe_layer_norm
torch.layer_norm = safe_layer_norm
