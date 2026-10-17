"""GEMMs with alpha == 0 or k == 0 through the SGEMM shim: BLAS defines the result as beta * C
without reading A or B, so an Inf or NaN in A or B must not reach C, and beta == 0 must write
zeros even over a NaN in C. The alpha == 0 cases reach all four intercepted entry points
(addmm and baddbmm, fp32 and fp16), one with a C wider than one launch's grid; PyTorch
resolves the k == 0 cases before calling BLAS, so those check only the end result.

Run on the GPU under test:  python3 -u shim_alpha0_check.py      Expect 15 of 15.
"""
import torch
torch.manual_seed(0)
bad = 0
def check(tag, got, want):
    global bad
    g, w = got.cpu(), want.cpu()
    ok = torch.equal(g.isnan(), w.isnan()) and torch.equal(g.nan_to_num(), w.nan_to_num())
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {tag}  nan={int(got.isnan().sum())}")
for dt in (torch.float32, torch.float16):
    for beta in (0.5, 0.0, 1.0):
        C = torch.randn(300, 200, device="cuda", dtype=dt); C[0, 0] = float("nan") if beta == 1.0 else C[0, 0]
        A = torch.randn(300, 150, device="cuda", dtype=dt); A[5, 7] = float("nan"); A[9, 1] = float("inf")
        B = torch.randn(150, 200, device="cuda", dtype=dt)
        want = torch.zeros_like(C) if beta == 0.0 else (C * beta if beta != 1.0 else C.clone())
        check(f"addmm {dt} alpha=0 beta={beta}", torch.addmm(C, A, B, beta=beta, alpha=0.0), want)
        Cb = torch.randn(4, 90, 70, device="cuda", dtype=dt)
        Ab = torch.randn(4, 90, 60, device="cuda", dtype=dt); Ab[2, 3, 4] = float("nan")
        Bb = torch.randn(4, 60, 70, device="cuda", dtype=dt)
        wantb = torch.zeros_like(Cb) if beta == 0.0 else (Cb * beta if beta != 1.0 else Cb.clone())
        check(f"baddbmm {dt} alpha=0 beta={beta}", torch.baddbmm(Cb, Ab, Bb, beta=beta, alpha=0.0), wantb)
    C = torch.randn(64, 48, device="cuda", dtype=dt)
    check(f"addmm {dt} k=0 beta=0.5", torch.addmm(C, torch.empty(64, 0, device="cuda", dtype=dt),
          torch.empty(0, 48, device="cuda", dtype=dt), beta=0.5, alpha=2.0), C * 0.5)
C = torch.randn(4, 70000, device="cuda"); A = torch.randn(4, 8, device="cuda"); A[0, 0] = float("nan")
check("addmm fp32 alpha=0 beta=0.25 N=70000", torch.addmm(C, A, torch.randn(8, 70000, device="cuda"), beta=0.25, alpha=0.0), C * 0.25)
print("RESULT:", "FAIL" if bad else "PASS", f"({bad} failing)")
