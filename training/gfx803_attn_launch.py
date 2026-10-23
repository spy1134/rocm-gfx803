"""Runs a training launcher with F.scaled_dot_product_attention patched to
gfx803's fused attention.

Usage: gfx803_attn_launch.py LAUNCHER [args...]

gfx803-train.sh runs sd-scripts with --sdpa, so every UNet attention goes
through F.scaled_dot_product_attention. PyTorch's own implementation keeps
the full score and probability matrices for the backward pass, 2.6 GB for
one SD1.5 self-attention at 512 px, which is what forces gradient
checkpointing on an 8 GB card. The fused kernels (/opt/gfx803-attn, built
from tools/gfx803_fused_attention.hip) recompute them from a per-row
log-sum-exp, so memory grows with the token count instead. Calls the
kernels do not cover (masks, dropout, head sizes other than 40/64/80) keep
PyTorch's implementation. GFX803_ATTN_TRAIN=0 sends every call needing
gradients back to PyTorch.
"""
import os
import runpy
import sys

ATTN_DIR = os.environ.get("GFX803_ATTN_DIR", "/opt/gfx803-attn")

try:
    sys.path.insert(0, ATTN_DIR)
    import gfx803_attention
    if gfx803_attention.is_available():
        gfx803_attention.patch_torch_sdpa()
    else:
        print("[gfx803-attn] fused attention library not loaded; PyTorch attention", file=sys.stderr)
except Exception as e:  # training still runs, on PyTorch's attention
    print(f"[gfx803-attn] could not load the fused attention ({e}); PyTorch attention", file=sys.stderr)

launcher = sys.argv[1]
sys.argv = [launcher] + sys.argv[2:]
runpy.run_path(launcher, run_name="__main__")
