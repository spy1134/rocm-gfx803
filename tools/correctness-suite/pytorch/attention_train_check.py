"""Training path of the gfx803 fused attention (tools/gfx803_attention.py): output and dQ/dK/dV of
F.scaled_dot_product_attention with gradients against float64 autograd on the CPU, at SD1.5/SDXL-sized
self- and cross-attention, ragged sizes, fp16 inputs and large logits; then checks that calls without
gradients still give the inference kernel's exact output.

Run on the GPU under test, with tools/ (built libgfx803_attn.so next to gfx803_attention.py) at /tools:
    python3 -u attention_train_check.py          Expect PASS.
"""
import os, sys, math, torch, torch.nn.functional as F
sys.path.insert(0, os.environ.get("GFX803_TOOLS", "/tools")); import gfx803_attention as ga
ga.patch_torch_sdpa()
torch.manual_seed(0); bad = 0
for (B, H, N, M, D, dt, mag) in [(1, 8, 2048, 2048, 40, torch.float32, 1), (2, 2, 1500, 1600, 64, torch.float32, 1),
                                 (1, 4, 1024, 2048, 80, torch.float32, 1), (1, 2, 1500, 1500, 40, torch.float16, 1),
                                 (1, 2, 1450, 1450, 64, torch.float32, 3),
                                 (2, 8, 4096, 77, 40, torch.float32, 1), (1, 8, 256, 256, 80, torch.float32, 1), (1, 2, 129, 33, 64, torch.float32, 1)]:
    q, k, v = (torch.randn(B, H, n, D, device="cuda", dtype=dt) * mag for n in (N, M, M)); g = torch.randn(B, H, N, D, device="cuda", dtype=dt)
    qq, kk, vv = (t.clone().requires_grad_() for t in (q, k, v))
    out = F.scaled_dot_product_attention(qq, kk, vv); out.backward(g)
    used = out.grad_fn.__class__.__name__
    qd, kd, vd = (t.double().cpu().requires_grad_() for t in (q, k, v))
    od = torch.softmax(qd @ kd.transpose(-1, -2) / math.sqrt(D), -1) @ vd; od.backward(g.double().cpu())
    errs = []
    for name, a, r in (("o", out, od.detach()), ("dq", qq.grad, qd.grad), ("dk", kk.grad, kd.grad), ("dv", vv.grad, vd.grad)):
        e = ((a.double().cpu() - r).abs().max() / r.abs().max()).item(); errs.append(f"{name} {e:.1e}")
        bad += e > (2e-3 if dt == torch.float16 else 1e-4)
    print(B, H, N, M, D, str(dt)[6:], used, " ".join(errs))
# inference path untouched: no grad -> fused forward, equal to the plain forward entry
q = torch.randn(2, 8, 4096, 40, device="cuda"); k = torch.randn(2, 8, 4096, 40, device="cuda"); v = torch.randn(2, 8, 4096, 40, device="cuda")
with torch.no_grad():
    a = F.scaled_dot_product_attention(q, k, v); b = ga.fused_attention(q, k, v)
print("inference unchanged:", torch.equal(a, b)); bad += not torch.equal(a, b)
print("RESULT:", "FAIL" if bad else "PASS")
