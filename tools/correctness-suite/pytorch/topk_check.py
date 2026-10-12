"""torch.topk on the GPU against the CPU, plus the samplers built on it.

Run only on an image with rocm6.4.4/patches/pytorch/gfx803-topk-scan-warp-size.patch: without it the first
case traps a wave and the driver resets the GPU.
Values must match exactly; indices must point at elements with the returned values (ties may pick
different, equally valid, indices).
"""
import torch

torch.manual_seed(0)
bad = cases = 0
sort_missing = []


def check(x, k, dim=-1, largest=True, sorted_=True, label=""):
    global bad, cases
    try:
        vg, ig = torch.topk(x.cuda(), k, dim=dim, largest=largest, sorted=sorted_)
        torch.cuda.synchronize()
    except torch.AcceleratorError as e:
        # Sorting a large k goes through torch.sort, whose radix-sort path has no gfx803 kernel for
        # some sizes (a separate bug, present before the scan patch). It errors cleanly; count it apart.
        if "invalid device function" not in str(e):
            raise
        sort_missing.append((tuple(x.shape), k, str(x.dtype)))
        return
    cases += 1
    vc, ic = torch.topk(x, k, dim=dim, largest=largest, sorted=sorted_)
    vg, ig = vg.cpu(), ig.cpu()
    if not sorted_:
        vg, order = vg.sort(dim=dim, descending=largest)
        ig = ig.gather(dim, order)
        vc = vc.sort(dim=dim, descending=largest)[0]
    ok = torch.equal(vg, vc) and torch.equal(x.gather(dim, ig), vg)
    if not ok:
        bad += 1
        if bad <= 10:
            print(f"BAD {label} shape {tuple(x.shape)} k {k} dim {dim} largest {largest} sorted {sorted_} {x.dtype}")


shapes = [(8194,), (1, 8194), (2, 8194), (64,), (65,), (1000,), (4096,), (151936,), (32, 1000), (4, 50000),
          (256, 256), (3, 7, 129), (1, 1 << 20)]
for shape in shapes:
    for dtype in (torch.float32, torch.float16, torch.int64):
        x = torch.randn(shape) * 10 if dtype != torch.int64 else torch.randint(-50, 50, shape)
        x = x.to(dtype)
        n = shape[-1]
        for k in sorted({1, 2, 5, 50, min(n, 1000), n // 2 or 1, n}):
            for largest in (True, False):
                check(x, k, largest=largest, sorted_=True, label="rand")
            check(x, k, largest=True, sorted_=False, label="unsorted")
        if len(shape) > 1:
            check(x, 1, dim=0, label="dim0")
# Heavy ties (logits after a min-p/top-p mask): many equal values, -inf runs.
for n in (8194, 4096, 100000):
    x = torch.randn(n)
    x[torch.rand(n) < 0.95] = float("-inf")
    for k in (1, 10, 50):
        check(x, k, label="masked")
    check(torch.zeros(n), 7, label="all-equal")
print(f"topk: {cases - bad}/{cases} cases match the CPU; {len(sort_missing)} more hit the missing sort kernel, e.g. {sort_missing[:3]}")

# The samplers that hit the bug, end to end: same logits, CPU-sampled from GPU-filtered probabilities.
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper, MinPLogitsWarper
bad_s = 0
for _ in range(50):
    logits = torch.randn(1, 8194) * 4
    for w in (TopKLogitsWarper(50), TopPLogitsWarper(0.9), MinPLogitsWarper(0.05)):
        ids = torch.zeros(1, 1, dtype=torch.long)
        g = w(ids.cuda(), logits.cuda()).cpu()
        c = w(ids, logits)
        bad_s += not torch.equal(torch.isinf(g), torch.isinf(c))
print(f"HF warpers: {150 - bad_s}/150 keep the same token set as the CPU")
