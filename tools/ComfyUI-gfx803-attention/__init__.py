"""
ComfyUI Custom Node for AMD gfx803 (Polaris) Accelerated Kernels.

Hooks:
1. F.scaled_dot_product_attention & ComfyUI attention_split / optimized_attention:
   Fused multi-wave online-softmax attention (D=40/64/80), reading fp16 in place.
   Saves 2+ GB transient VRAM allocations and provides up to 2.2x speedup.
2. ComfyUI GEGLU forward pass:
   Native 128-bit vectorized exact erff GEGLU activation kernel (>180 GB/s bandwidth).
3. torch.nn.functional.group_norm and torch.nn.GroupNorm:
   Native 128-bit vectorized GroupNorm kernel (up to 1.8x speedup).
"""

import os
import sys

import torch

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

try:
    import gfx803_attention

    if gfx803_attention.is_available():
        # 1. Patch stock PyTorch SDPA
        gfx803_attention.patch_torch_sdpa()

        # 2. Patch ComfyUI internal attention if comfy is imported
        try:
            import comfy.ldm.modules.attention as comfy_attn

            _orig_split = comfy_attn.attention_split

            @comfy_attn.wrap_attn
            def patched_attention_split(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
                if skip_reshape:
                    b, _, N_q, dim_head = q.shape
                    N_kv = k.shape[2]
                else:
                    b, N_q, total_dim = q.shape
                    N_kv = k.shape[1]
                    dim_head = total_dim // heads

                # Eligible for gfx803 fused attention:
                # CUDA, unmasked, forward only (no autograd), D in (40, 64, 80), no GQA.
                # D=160 cross-attention regresses to 0.76x on Polaris hardware (0.140 ms vs 0.106 ms stock)
                # because short sequence length (N_kv <= 128) has low memory overhead while large head
                # dimension reduces CU occupancy. Stock PyTorch Math Attention is faster for D=160.
                if (
                    gfx803_attention.is_available()
                    and q.is_cuda
                    and mask is None
                    and not (q.requires_grad or k.requires_grad or v.requires_grad)
                    and (dim_head in (40, 64, 80))
                    and not kwargs.get("enable_gqa", False)
                ):
                    scale = kwargs.get("scale", dim_head ** -0.5)

                    if not skip_reshape:
                        q_4d = q.view(b, N_q, heads, dim_head).transpose(1, 2)
                        k_4d = k.view(b, N_kv, heads, dim_head).transpose(1, 2)
                        v_4d = v.view(b, N_kv, heads, dim_head).transpose(1, 2)
                    else:
                        q_4d, k_4d, v_4d = q, k, v
                    # The output goes straight into the layout this function returns.
                    if not skip_output_reshape:
                        out = torch.empty((b, N_q, heads * dim_head), device=q.device, dtype=q.dtype)
                        out_4d = out.view(b, N_q, heads, dim_head).transpose(1, 2)
                    else:
                        out = out_4d = torch.empty((b, heads, N_q, dim_head), device=q.device, dtype=q.dtype)
                    if gfx803_attention.fused_attention_into(q_4d, k_4d, v_4d, out_4d, scale):
                        return out

                    out_4d = gfx803_attention.fused_attention(q_4d, k_4d, v_4d, scale=scale)

                    if not skip_output_reshape:
                        return out_4d.transpose(1, 2).reshape(b, N_q, heads * dim_head)
                    else:
                        return out_4d

                return _orig_split(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

            # Hook into comfy_attn globals
            comfy_attn.attention_split = patched_attention_split
            if hasattr(comfy_attn, "optimized_attention"):
                comfy_attn.optimized_attention = patched_attention_split
            if hasattr(comfy_attn, "optimized_attention_masked"):
                comfy_attn.optimized_attention_masked = patched_attention_split

            # Hook ComfyUI GEGLU
            if hasattr(comfy_attn, "GEGLU"):
                import gfx803_geglu
                _orig_geglu_forward = comfy_attn.GEGLU.forward
                def patched_geglu_forward(self, x):
                    # The kernel's output has no grad_fn, so any trainable input, including the projection's own
                    # parameters, must take the stock path.
                    if (gfx803_geglu._LIB is not None and x.is_cuda and not x.requires_grad
                            and not any(p.requires_grad for p in self.proj.parameters())):
                        proj_out = self.proj(x)
                        return gfx803_geglu.gfx803_geglu(proj_out, approximate_tanh=False)
                    return _orig_geglu_forward(self, x)
                comfy_attn.GEGLU.forward = patched_geglu_forward
                print("[ComfyUI-gfx803-attention] Hooked ComfyUI GEGLU forward pass (exact erf).")

            print("[ComfyUI-gfx803-attention] Hooked ComfyUI attention_split & optimized_attention (D=40/64/80).")
        except Exception as e_comfy:
            print(f"[ComfyUI-gfx803-attention] Note: ComfyUI internal attention hooking skipped ({e_comfy})")

        # 3. Patch GroupNorm in PyTorch and ComfyUI
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import gfx803_norm
            if gfx803_norm._LIB is not None:
                # ComfyUI's comfy.ops.disable_weight_init.GroupNorm overrides forward()
                # and invokes F.group_norm directly with cast weights, bypassing nn.GroupNorm.forward.
                # Patching F.group_norm ensures both ComfyUI models and standard modules are accelerated.
                _orig_f_group_norm = F.group_norm

                def patched_f_group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
                    # apply_silu=False is intentional: UNet ResBlocks execute SiLU as a separate
                    # subsequent layer in nn.Sequential; passing True would evaluate SiLU twice.
                    if (
                        input.is_cuda
                        and input.dtype == torch.float32
                        and not (input.requires_grad or (weight is not None and weight.requires_grad) or (bias is not None and bias.requires_grad))
                        and input.dim() in (2, 3, 4)
                        and input.shape[1] % num_groups == 0
                        and (input.shape[1] // num_groups) <= 128
                    ):
                        return gfx803_norm.gfx803_group_norm(input, num_groups, weight, bias, eps, apply_silu=False)
                    return _orig_f_group_norm(input, num_groups, weight, bias, eps)

                F.group_norm = patched_f_group_norm

                _orig_gn_forward = nn.GroupNorm.forward
                def patched_gn_forward(self, input):
                    if (input.is_cuda and input.dtype == torch.float32 and 
                        not (input.requires_grad or (self.weight is not None and self.weight.requires_grad) or (self.bias is not None and self.bias.requires_grad))):
                        return gfx803_norm.gfx803_group_norm(input, self.num_groups, self.weight, self.bias, self.eps, apply_silu=False)
                    return _orig_gn_forward(self, input)
                nn.GroupNorm.forward = patched_gn_forward

                print("[ComfyUI-gfx803-attention] Hooked torch.nn.functional.group_norm & GroupNorm.forward.")
        except Exception as e_gn:
            print(f"[ComfyUI-gfx803-attention] Note: GroupNorm hooking skipped ({e_gn})")

    else:
        print("[ComfyUI-gfx803-attention] Native library not found or GPU not available.")
except Exception as e:
    print(f"[ComfyUI-gfx803-attention] Failed to initialize: {e}")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
