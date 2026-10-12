"""Torch ops built on rocPRIM (select, scan, radix/merge sort, unique, reduce-by-key) on the GPU against the CPU.

Without rocm6.4.4/patches/pytorch/gfx803-rocprim-private-namespace.patch most of these fail to launch on gfx803
("invalid device function"); with it every case must launch and match.
"""
import torch

torch.manual_seed(0)
results = {}


def case(name, fn, cmp=torch.equal):
    try:
        g = fn("cuda")
        torch.cuda.synchronize()
    except Exception as e:
        results.setdefault(name, []).append(f"ERROR {str(e).splitlines()[0][:60]}")
        return
    c = fn("cpu")
    g = g if isinstance(g, tuple) else (g,)
    c = c if isinstance(c, tuple) else (c,)
    ok = all(cmp(a.cpu(), b) for a, b in zip(g, c))
    results.setdefault(name, []).append("ok" if ok else "MISMATCH")


def seeded(device, make):
    torch.manual_seed(1)
    return make().to(device)


for n in (1, 10, 1000, 4097, 100000, 3_000_000):
    mask = lambda: torch.rand(n) > 0.5
    vals = lambda: torch.randn(n)
    ints = lambda: torch.randint(0, 997, (n,))
    case("nonzero", lambda d: seeded(d, mask).nonzero())
    case("masked_select", lambda d: seeded(d, vals)[seeded(d, mask)])
    case("masked assign", lambda d: seeded(d, vals).masked_fill(seeded(d, mask), 3.0))
    case("unique", lambda d: torch.unique(seeded(d, ints), return_inverse=True, return_counts=True))
    case("unique_consecutive", lambda d: torch.unique_consecutive(seeded(d, ints).sort()[0], return_counts=True))
    for dt in (torch.float32, torch.float16, torch.int64):
        mk = (lambda: torch.randn(n).to(dt)) if dt != torch.int64 else ints
        case(f"sort {dt}", lambda d: seeded(d, mk).sort(stable=True))
        case(f"sort desc {dt}", lambda d: seeded(d, mk).sort(descending=True, stable=True))
        case(f"argsort {dt}", lambda d: seeded(d, mk).argsort(stable=True))
    case("cumsum int", lambda d: seeded(d, ints).cumsum(0))
    case("topk k=n/2", lambda d: seeded(d, vals).topk(max(1, n // 2))[0])
    case("index_put accumulate", lambda d: torch.zeros(1000, device=d).index_put_((seeded(d, ints) % 1000,), seeded(d, vals), accumulate=True),
         cmp=lambda a, b: torch.allclose(a, b, rtol=1e-4, atol=1e-3))
    case("bincount", lambda d: torch.bincount(seeded(d, ints)))
    # linspace rounds its last bit differently on the GPU and the CPU; build the queries once.
    case("searchsorted", lambda d: torch.searchsorted(seeded(d, vals).sort()[0], torch.linspace(-3, 3, 101).to(d)))
case("sort 2D rows", lambda d: seeded(d, lambda: torch.randn(64, 5000)).sort(dim=1, stable=True))
case("sort 2D cols", lambda d: seeded(d, lambda: torch.randn(5000, 64)).sort(dim=0, stable=True))
case("randperm is a permutation", lambda d: torch.randperm(100000, device=d).sort()[0])
w = torch.nn.Embedding(1000, 16)
def emb_grad(d):
    m = torch.nn.Embedding(1000, 16).to(d)
    m.weight.data.copy_(w.weight.data)
    idx = seeded(d, lambda: torch.randint(0, 1000, (50000,)))
    m(idx).sum().backward()
    return m.weight.grad
case("embedding backward", emb_grad, cmp=lambda a, b: torch.allclose(a, b, rtol=1e-4, atol=1e-3))

bad = 0
for name, rs in results.items():
    good = rs.count("ok")
    bad += len(rs) - good
    print(f"{name:28s} {good}/{len(rs)}" + ("" if good == len(rs) else f"  {sorted(set(r for r in rs if r != 'ok'))}"))
print(f"TOTAL: {sum(len(r) for r in results.values()) - bad}/{sum(len(r) for r in results.values())} ok")
