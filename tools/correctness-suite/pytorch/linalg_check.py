"""torch.linalg on the GPU against float64 on the CPU.

hipSOLVER and rocSOLVER carry the GPU work. A rocSOLVER built without a gfx803 code object does not fail:
its kernel launches are never checked, so the LU factors come back unchanged, with stale pivots and info,
and later calls index with those pivots and fault. The first block therefore factors one tiny matrix and
leaves before any other call when the factors are wrong.

Every result is judged by a residual (P@L@U against A, A@x against B, Q@R against A, ...), because factor
signs and eigenvector directions are not unique. Residuals are relative to the size of the inputs.
Exit status is 1 when any case fails.
"""
import os
import sys
import torch

torch.manual_seed(0)
dev = torch.device("cuda")
TOL = {torch.float32: 2e-4, torch.float64: 1e-9}
# LINALG_MAX_N caps the sizes, for a quick run.
SIZES = tuple(n for n in (2, 3, 5, 16, 33, 64, 129, 257) if n <= int(os.environ.get("LINALG_MAX_N", "257")))
BATCHES = (None, 4)
bad = cases = 0
skipped = []


def rel(a, ref_scale):
    return (a.abs().max() / ref_scale).item()


def run(name, n, batch, dtype, fn):
    global bad, cases
    cases += 1
    try:
        err = fn()
        torch.cuda.synchronize()
    except Exception as e:  # a wrong factorization raises inside torch for singular-looking input
        if "MAGMA" in str(e):
            # Cholesky and least squares on a ROCm device go to MAGMA, which this PyTorch build does not
            # include; torch refuses the call, so there is no result to judge.
            skipped.append(name)
            cases -= 1
            return
        err = float("inf")
        msg = str(e).splitlines()[0][:100]
    else:
        msg = ""
    if not err <= TOL[dtype]:
        bad += 1
        if bad <= 25:
            print(f"BAD {name} n={n} batch={batch} {dtype}: residual {err:.3e} (limit {TOL[dtype]:.0e}) {msg}")


def mk(shape, dtype, spd=False, dominant=False):
    a = torch.randn(*shape, dtype=torch.float64)
    n = shape[-1]
    if spd:
        a = a @ a.transpose(-1, -2) + n * torch.eye(n, dtype=torch.float64)
    elif dominant:
        a = a + n * torch.eye(n, dtype=torch.float64)
    return a.to(dtype)


# Preflight: one 4x4 factorization with a known answer.
a = mk((4, 4), torch.float32, dominant=True).to(dev)
LU, piv, info = torch.linalg.lu_factor_ex(a)
torch.cuda.synchronize()
info_ok = int(info.item()) == 0
piv_ok = bool(((piv >= 1) & (piv <= 4)).all().item())
unchanged = torch.equal(LU, a)
if not (info_ok and piv_ok) or unchanged:
    print(f"PREFLIGHT FAILED: info={int(info.item())} pivots={piv.tolist()} factors_equal_input={unchanged}")
    print("The GPU linear-algebra library returned no factorization; skipping every other case.")
    sys.exit(1)
P, L, U = torch.lu_unpack(LU, piv)
if not (P @ L @ U - a).abs().max().item() < 1e-4 * a.abs().max().item():
    print("PREFLIGHT FAILED: P@L@U does not reproduce the input")
    sys.exit(1)

for dtype in (torch.float32, torch.float64):
    for n in SIZES:
        for batch in BATCHES:
            shape = (n, n) if batch is None else (batch, n, n)
            rshape = (n, 3) if batch is None else (batch, n, 3)

            A = mk(shape, dtype, dominant=True)
            Ag = A.to(dev)
            scale = A.abs().max().double()

            def lu():
                LU, piv, info = torch.linalg.lu_factor_ex(Ag)
                if int(info.abs().max().item()) != 0:
                    return float("inf")
                # Unpack and multiply on the CPU so that the residual does not depend on GPU matmul.
                P, L, U = torch.lu_unpack(LU.cpu().double(), piv.cpu())
                return rel(P @ L @ U - A.double(), scale)
            run("lu_factor", n, batch, dtype, lu)

            B = mk(rshape, dtype)
            Bg = B.to(dev)

            def solve():
                x = torch.linalg.solve(Ag, Bg).cpu().double()
                r = A.double() @ x - B.double()
                return rel(r, scale * x.abs().max() + B.abs().max().double())
            run("solve", n, batch, dtype, solve)

            def inv():
                Ai = torch.linalg.inv(Ag).cpu().double()
                eye = torch.eye(n, dtype=torch.float64)
                return rel(A.double() @ Ai - eye, torch.tensor(1.0, dtype=torch.float64) * n)
            run("inv", n, batch, dtype, inv)

            def det():
                d = torch.linalg.det(Ag).cpu().double()
                ref = torch.linalg.det(A.double())
                return ((d - ref).abs() / ref.abs()).max().item()
            if n <= 16:  # the determinant of a larger dominant matrix overflows float32
                run("det", n, batch, dtype, det)

            S = mk(shape, dtype, spd=True)
            Sg = S.to(dev)
            sscale = S.abs().max().double()

            def chol():
                Lc = torch.linalg.cholesky(Sg).cpu().double()
                return rel(Lc @ Lc.transpose(-1, -2) - S.double(), sscale)
            run("cholesky", n, batch, dtype, chol)

            def eigh():
                w, v = torch.linalg.eigh(Sg)
                w, v = w.cpu().double(), v.cpu().double()
                res = rel(S.double() @ v - v * w.unsqueeze(-2), sscale)
                ref = torch.linalg.eigvalsh(S.double())
                return max(res, rel(w - ref, sscale))
            run("eigh", n, batch, dtype, eigh)

            m = n + 5
            G = mk((m, n) if batch is None else (batch, m, n), dtype)
            Gg = G.to(dev)
            gscale = G.abs().max().double()

            def qr():
                Q, R = torch.linalg.qr(Gg)
                Q, R = Q.cpu().double(), R.cpu().double()
                eye = torch.eye(n, dtype=torch.float64)
                return max(rel(Q @ R - G.double(), gscale), rel(Q.transpose(-1, -2) @ Q - eye, torch.tensor(1.0, dtype=torch.float64)))
            run("qr", n, batch, dtype, qr)

            def svd():
                U_, s, Vh = torch.linalg.svd(Gg, full_matrices=False)
                U_, s, Vh = U_.cpu().double(), s.cpu().double(), Vh.cpu().double()
                res = rel(U_ @ torch.diag_embed(s) @ Vh - G.double(), gscale)
                ref = torch.linalg.svdvals(G.double())
                return max(res, rel(s - ref, gscale))
            if n <= 129:
                run("svd", n, batch, dtype, svd)

            def lstsq():
                rhs = mk((m, 2) if batch is None else (batch, m, 2), dtype)
                sol = torch.linalg.lstsq(Gg, rhs.to(dev)).solution.cpu().double()
                ref = torch.linalg.lstsq(G.double(), rhs.double()).solution
                return rel(sol - ref, ref.abs().max())
            if n <= 129:
                run("lstsq", n, batch, dtype, lstsq)

print(f"linalg: {cases - bad}/{cases} cases match the float64 CPU result; {len(skipped)} skipped for lack of MAGMA ({sorted(set(skipped))})")
sys.exit(1 if bad else 0)
