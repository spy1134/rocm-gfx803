"""torchvision ops on the GPU against the same ops on the CPU: roi_align / roi_pool / ps_roi_align forward and
backward, deform_conv2d forward and backward, and nms (through torch.ops, the kernel torchvision.ops.nms dispatches to)."""
import sys, torch, torchvision
from torchvision import ops
torch.manual_seed(0)
dev = "cuda"
bad = 0
def chk(name, f, tol=1e-4):
    global bad
    try:
        a, b = f(dev), f("cpu")
    except Exception as e:
        print("ERROR", name, str(e).splitlines()[0][:100]); bad += 1; return
    errs = [float((x.cpu().double() - y.double()).abs().max()) for x, y in zip(a, b)]
    ok = max(errs) <= tol
    bad += not ok
    print("ok   " if ok else "WRONG", name, " ".join(f"{e:.2e}" for e in errs))
def mk(shape, d, rg=False):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(sum(shape))).to(d).requires_grad_(rg)
def boxes(d, n=20, h=32, w=32):
    g = torch.Generator().manual_seed(5)
    xy = torch.rand(n, 2, generator=g) * (w / 2); wh = torch.rand(n, 2, generator=g) * (w / 2) + 2
    b = torch.cat([torch.zeros(n, 1), xy, xy + wh], 1)
    return b.to(d)
def roi(op, **kw):
    def f(d):
        x = mk((2, 8, 32, 32), d, True); b = boxes(d); b[:10, 0] = 1
        out = op(x, b, **kw); out.sum().backward(); return [out.detach(), x.grad]
    return f
chk("roi_align", roi(ops.roi_align, output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2))
chk("roi_pool", roi(ops.roi_pool, output_size=(7, 7), spatial_scale=1.0))
def psr(d):
    x = mk((2, 4 * 9, 32, 32), d, True); b = boxes(d); b[:10, 0] = 1
    out = ops.ps_roi_align(x, b, output_size=(3, 3)); out.sum().backward(); return [out.detach(), x.grad]
chk("ps_roi_align", psr)
def dcn(d):
    x = mk((2, 8, 16, 16), d, True); w = mk((6, 8, 3, 3), d, True); off = (mk((2, 18, 16, 16), d) * 0.5).requires_grad_(True)
    out = ops.deform_conv2d(x, off, w, padding=1); out.sum().backward(); return [out.detach(), x.grad, w.grad, off.grad]
chk("deform_conv2d", dcn, 1e-3)
def nmsf(d):
    b = boxes(d, 300) [:, 1:]; s = torch.rand(300, generator=torch.Generator().manual_seed(9)).to(d)
    return [torch.ops.torchvision.nms(b, s, 0.5).sort()[0].double()]
chk("nms (torch.ops)", nmsf, 0.0)
print("RESULT tv_check bad=%d" % bad); sys.exit(1 if bad else 0)
