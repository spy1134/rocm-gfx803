#!/usr/bin/env python3
"""Layer probe (a): rocBLAS routines that rocSOLVER depends on, called directly, fp32, against float64 numpy.

It launches kernels on the card; run it with the display idle or on a card that drives no monitor.
Sizes ascend, one line per case, and the probe stops at the first wrong case. No pivots or indices produced by
the device are ever fed back into another device call, so a wrong result cannot become an out-of-range access.

Why these calls: rocSOLVER never calls rocblas_sgemm / rocblas_gemm_ex (the entry points the LD_PRELOAD shim
replaces). It imports rocblas_internal_*_template, and hipBLAS's strsm (torch.linalg.solve_triangular, which
torch.linalg.solve uses) reaches Tensile through trtri + gemm inside librocblas.so. gemm and trsm below therefore
run the stock Tensile fp32 kernels; each gemm shape is repeated because a Tensile fault shows as "first call
right, later calls wrong".

Usage (inside rocm-gfx803:rocm6.4.4 with --device flags):
  python3 -u probe_rocblas.py [--max-n 64] [--ops regex] [--reps 6]
Controls: NOFREE=1 turns hipFree into a no-op, which passes every case on the unpatched runtime;
CLR_GFX8_TC_INVALIDATE=0 switches the libamdhip64 fix off in the patched one, and the run then stops at case 225.
PAD_BYTES and PAD_FILL (zero, nan, ff) surround each buffer with known bytes to tell a stale read from an out-of-bounds one.
"""
import argparse
import re
import sys

import numpy as np

from _hip import *  # noqa: F401,F403

ap = argparse.ArgumentParser()
ap.add_argument("--max-n", type=int, default=64)
ap.add_argument("--ops", default="")
ap.add_argument("--reps", type=int, default=6)
args = ap.parse_args()
SIZES = [n for n in (2, 3, 5, 8, 16, 17, 32, 33, 64, 65, 128) if n <= args.max_n]
want = re.compile(args.ops) if args.ops else None

header("probe_rocblas")
h = make_handle()
r = rng(1234)
one, mone, zero = c_f(1.0), c_f(-1.0), c_f(0.0)


def f32(*shape):
    return r.standard_normal(shape).astype(np.float32)


def dominant(n, lower=None, unit=False):
    a = f32(n, n) * 0.3 + n * np.eye(n, dtype=np.float32)
    if lower is True:
        a = np.tril(a)
    elif lower is False:
        a = np.triu(a)
    if unit:
        a = a - np.diag(np.diag(a)) + np.eye(n, dtype=np.float32)
    return a.astype(np.float32)


def run(op, shape, fn):
    if want and not want.search(op):
        return
    err, tol, extra = fn()
    report(op, shape, err, tol, extra)


# Preflight: a four-element scal must change the data before anything bigger runs.
x0 = np.arange(1, 5, dtype=np.float32)
dx = Dev(x0)
st(rb.rocblas_sscal(h, c_i(4), ctypes.byref(c_f(2.0)), dx.p, c_i(1)), "preflight sscal")
sync("preflight sscal")
if not np.array_equal(dx.get(), 2 * x0):
    print("PREFLIGHT FAILED: sscal did not scale four elements; no rocBLAS kernel is running correctly")
    sys.exit(1)
print("preflight ok")
dx.free()

for n in SIZES:
    # ---- level 1
    for inc in (1, 2):
        x = f32(n * inc)
        dx = Dev(x)
        res = c_i(0)

        def f_iamax():
            st(rb.rocblas_isamax(h, c_i(n), dx.p, c_i(inc), ctypes.byref(res)), "isamax")
            sync("isamax")
            ref = int(np.argmax(np.abs(x[::inc][:n]))) + 1
            return (0.0 if res.value == ref else 1.0), 0.5, f"got={res.value} ref={ref}"
        run("isamax", f"n={n} inc={inc}", f_iamax)

        def f_nrm2():
            o = c_f(0)
            st(rb.rocblas_snrm2(h, c_i(n), dx.p, c_i(inc), ctypes.byref(o)), "snrm2")
            sync("snrm2")
            return relerr(o.value, np.linalg.norm(x[::inc][:n].astype(np.float64))), 1e-5, ""
        run("snrm2", f"n={n} inc={inc}", f_nrm2)

        y = f32(n * inc)
        dy = Dev(y)

        def f_dot():
            o = c_f(0)
            st(rb.rocblas_sdot(h, c_i(n), dx.p, c_i(inc), dy.p, c_i(inc), ctypes.byref(o)), "sdot")
            sync("sdot")
            ref = float(np.dot(x[::inc][:n].astype(np.float64), y[::inc][:n].astype(np.float64)))
            return abs(o.value - ref) / (np.abs(x).max() * np.abs(y).max() * n), 1e-6, ""
        run("sdot", f"n={n} inc={inc}", f_dot)

        def f_scal():
            dx.put(x)
            st(rb.rocblas_sscal(h, c_i(n), ctypes.byref(c_f(0.37)), dx.p, c_i(inc)), "sscal")
            sync("sscal")
            ref = x.astype(np.float64).copy()
            ref[::inc][:n] *= np.float32(0.37)
            return relerr(dx.get(), ref), 1e-6, ""
        run("sscal", f"n={n} inc={inc}", f_scal)
        dx.free()
        dy.free()

    # ---- level 2
    m = n + 5
    A = f32(m, n)
    dA = Dev(A)
    for tr, name in ((OP_N, "N"), (OP_T, "T")):
        lx, ly = (n, m) if tr == OP_N else (m, n)
        x, y = f32(lx), f32(ly)

        def f_gemv():
            dx_, dy_ = Dev(x), Dev(y)
            st(rb.rocblas_sgemv(h, c_i(tr), c_i(m), c_i(n), ctypes.byref(c_f(-1.0)), dA.p, c_i(m), dx_.p, c_i(1),
                                ctypes.byref(c_f(1.0)), dy_.p, c_i(1)), "sgemv")
            sync("sgemv")
            Ad = A.astype(np.float64)
            ref = y.astype(np.float64) - (Ad @ x if tr == OP_N else Ad.T @ x)
            e = relerr(dy_.get(), ref)
            dx_.free(), dy_.free()
            return e, 1e-5, ""
        run(f"sgemv_{name}", f"m={m} n={n}", f_gemv)

    def f_ger():
        x, y = f32(m), f32(n)
        dx_, dy_, dG = Dev(x), Dev(y), Dev(A)
        st(rb.rocblas_sger(h, c_i(m), c_i(n), ctypes.byref(c_f(-1.0)), dx_.p, c_i(1), dy_.p, c_i(1), dG.p, c_i(m)),
           "sger")
        sync("sger")
        ref = A.astype(np.float64) - np.outer(x.astype(np.float64), y.astype(np.float64))
        e = relerr(dG.get(), ref)
        dx_.free(), dy_.free(), dG.free()
        return e, 1e-5, ""
    run("sger", f"m={m} n={n}", f_ger)
    dA.free()

    for uplo, un in ((LOWER, "L"), (UPPER, "U")):
        for diag, dn in ((UNIT, "unit"), (NONUNIT, "nonunit")):
            for tr, tn in ((OP_N, "N"), (OP_T, "T")):
                T = dominant(n, lower=(uplo == LOWER), unit=(diag == UNIT))
                x = f32(n)

                def f_trmv():
                    dT, dx_ = Dev(T), Dev(x)
                    st(rb.rocblas_strmv(h, c_i(uplo), c_i(tr), c_i(diag), c_i(n), dT.p, c_i(n), dx_.p, c_i(1)),
                       "strmv")
                    sync("strmv")
                    Td = T.astype(np.float64)
                    Tm = np.tril(Td) if uplo == LOWER else np.triu(Td)
                    if diag == UNIT:
                        Tm = Tm - np.diag(np.diag(Tm)) + np.eye(n)
                    ref = (Tm if tr == OP_N else Tm.T) @ x.astype(np.float64)
                    e = relerr(dx_.get(), ref)
                    dT.free(), dx_.free()
                    return e, 1e-5, ""
                run(f"strmv_{un}{tn}_{dn}", f"n={n}", f_trmv)

                def f_trsv():
                    dT, dx_ = Dev(T), Dev(x)
                    st(rb.rocblas_strsv(h, c_i(uplo), c_i(tr), c_i(diag), c_i(n), dT.p, c_i(n), dx_.p, c_i(1)),
                       "strsv")
                    sync("strsv")
                    Td = T.astype(np.float64)
                    Tm = np.tril(Td) if uplo == LOWER else np.triu(Td)
                    if diag == UNIT:
                        Tm = Tm - np.diag(np.diag(Tm)) + np.eye(n)
                    ref = np.linalg.solve(Tm if tr == OP_N else Tm.T, x.astype(np.float64))
                    e = relerr(dx_.get(), ref)
                    dT.free(), dx_.free()
                    return e, 2e-4, ""
                run(f"strsv_{un}{tn}_{dn}", f"n={n}", f_trsv)

    # ---- level 3: trsm in the two forms torch.linalg.solve uses (L unit, then U non-unit), left side, no transpose
    for nrhs in sorted({3, n}):
        for uplo, diag, un in ((LOWER, UNIT, "L_unit"), (UPPER, NONUNIT, "U_nonunit")):
            T = dominant(n, lower=(uplo == LOWER), unit=(diag == UNIT))
            B = f32(n, nrhs)

            def f_trsm():
                dT, dB = Dev(T), Dev(B)
                st(rb.rocblas_strsm(h, c_i(LEFT), c_i(uplo), c_i(OP_N), c_i(diag), c_i(n), c_i(nrhs),
                                    ctypes.byref(c_f(1.0)), dT.p, c_i(n), dB.p, c_i(n)), "strsm")
                sync("strsm")
                Td = T.astype(np.float64)
                Tm = np.tril(Td) if uplo == LOWER else np.triu(Td)
                if diag == UNIT:
                    Tm = Tm - np.diag(np.diag(Tm)) + np.eye(n)
                e = relerr(dB.get(), np.linalg.solve(Tm, B.astype(np.float64)))
                dT.free(), dB.free()
                return e, 2e-4, ""
            run(f"strsm_{un}", f"n={n} nrhs={nrhs}", f_trsm)

    # gemm shapes of the kinds rocSOLVER issues: square update, tall-skinny panel update, thin rhs update
    jb = min(n, 32)
    shapes = [(n, n, n), (n + 3, n, jb), (n, 3, n), (max(n - jb, 1), n, jb)]
    for (gm, gn, gk) in shapes:
        A_, B_ = f32(gm, gk), f32(gk, gn)
        C_ = f32(gm, gn)
        for rep in range(args.reps):
            def f_gemm():
                dA_, dB_, dC_ = Dev(A_), Dev(B_), Dev(C_)
                st(rb.rocblas_sgemm(h, c_i(OP_N), c_i(OP_N), c_i(gm), c_i(gn), c_i(gk), ctypes.byref(c_f(-1.0)),
                                    dA_.p, c_i(gm), dB_.p, c_i(gk), ctypes.byref(c_f(1.0)), dC_.p, c_i(gm)), "sgemm")
                sync("sgemm")
                ref = C_.astype(np.float64) - A_.astype(np.float64) @ B_.astype(np.float64)
                e = relerr(dC_.get(), ref)
                dA_.free(), dB_.free(), dC_.free()
                return e, 1e-5 * max(1, gk // 8), f"rep={rep}"
            run("sgemm_NN(-1,+1)", f"m={gm} n={gn} k={gk}", f_gemm)

    # strided and pointer-array batched gemm (batch of 4), the shapes rocSOLVER's batched paths use
    bm, bn, bk, nb = n, n, jb, 4
    Ab, Bb, Cb = f32(bm, bk, nb), f32(bk, bn, nb), f32(bm, bn, nb)

    def f_sb():
        dA_, dB_, dC_ = Dev(Ab), Dev(Bb), Dev(Cb)
        st(rb.rocblas_sgemm_strided_batched(h, c_i(OP_N), c_i(OP_N), c_i(bm), c_i(bn), c_i(bk),
                                            ctypes.byref(c_f(-1.0)), dA_.p, c_i(bm), c_l(bm * bk), dB_.p, c_i(bk),
                                            c_l(bk * bn), ctypes.byref(c_f(1.0)), dC_.p, c_i(bm), c_l(bm * bn),
                                            c_i(nb)), "sgemm_strided_batched")
        sync("sgemm_strided_batched")
        ref = Cb.astype(np.float64) - np.einsum("ikb,kjb->ijb", Ab.astype(np.float64), Bb.astype(np.float64))
        e = relerr(dC_.get(), ref)
        dA_.free(), dB_.free(), dC_.free()
        return e, 1e-5 * max(1, bk // 8), ""
    run("sgemm_strided_batched", f"m={bm} n={bn} k={bk} b={nb}", f_sb)

    def f_pb():
        da = [Dev(np.ascontiguousarray(Ab[:, :, i])) for i in range(nb)]
        db = [Dev(np.ascontiguousarray(Bb[:, :, i])) for i in range(nb)]
        dc = [Dev(np.ascontiguousarray(Cb[:, :, i])) for i in range(nb)]
        ptrs = [Dev(np.array([d.p.value for d in grp], dtype=np.uint64)) for grp in (da, db, dc)]
        st(rb.rocblas_sgemm_batched(h, c_i(OP_N), c_i(OP_N), c_i(bm), c_i(bn), c_i(bk), ctypes.byref(c_f(-1.0)),
                                    ptrs[0].p, c_i(bm), ptrs[1].p, c_i(bk), ctypes.byref(c_f(1.0)), ptrs[2].p,
                                    c_i(bm), c_i(nb)), "sgemm_batched")
        sync("sgemm_batched")
        e = 0.0
        for i in range(nb):
            ref = Cb[:, :, i].astype(np.float64) - Ab[:, :, i].astype(np.float64) @ Bb[:, :, i].astype(np.float64)
            e = max(e, relerr(dc[i].get(), ref))
        for d in da + db + dc + ptrs:
            d.free()
        return e, 1e-5 * max(1, bk // 8), ""
    run("sgemm_batched(ptr array)", f"m={bm} n={bn} k={bk} b={nb}", f_pb)

print("ALL OK up to n=%d" % SIZES[-1])
