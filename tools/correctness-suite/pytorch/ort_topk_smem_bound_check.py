#!/usr/bin/env python3
"""Boundary test for topk-safe-small-k-smem-bound.patch, run inside rocm-gfx803:rocm6.4.4.

TopK is built as a raw ONNX protobuf because the image has onnxruntime but not the
onnx package. Every case checks the ROCm EP is the one running, then compares against
a numpy reference:
  sorted=1: values equal the reference top-K values in order, indices point at them.
  sorted=0: the multiset of returned values equals the reference top-K values, and
            indices are unique, in range and point at the returned values.
Inputs are distinct, so the index set is fully determined.

SafeSmallKTopK smem_bytes = 256 * (sizeof(T) + 8) + 8 * K exceeds 65536 at
K > 7808 (float) and K > 7872 (fp16). Cases sit on both sides of those bounds.
The cases just below the bound run SafeSmallKTopK with K near 7800, which is quadratic
in K per block and can take tens of seconds; the card drives the desktop, so run those
(--include-below) only while nothing else uses the display, and expect to wait.
"""
import argparse
import struct
import sys
import time

import numpy as np
import onnxruntime as ort


def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def fld(num, wire, payload):
    tag = varint((num << 3) | wire)
    if wire == 0:
        return tag + varint(payload)
    return tag + varint(len(payload)) + payload


def s(num, text):
    return fld(num, 2, text.encode())


def value_info(name, elem_type, dims):
    shape = b"".join(fld(1, 2, fld(1, 0, d)) for d in dims)
    tensor = fld(1, 0, elem_type) + fld(2, 2, shape)
    return s(1, name) + fld(2, 2, fld(1, 2, tensor))


def attr_int(name, v):
    return s(1, name) + fld(3, 0, v) + fld(20, 0, 2)


def topk_model(dtype, n, dim, k, largest, sorted_):
    elem = {"float32": 1, "float16": 10}[dtype]
    k_tensor = fld(1, 0, 1) + fld(2, 0, 7) + s(8, "K") + fld(9, 2, struct.pack("<q", k))
    node = (s(1, "X") + s(1, "K") + s(2, "V") + s(2, "I") + s(3, "topk") + s(4, "TopK")
            + fld(5, 2, attr_int("axis", 1)) + fld(5, 2, attr_int("largest", largest))
            + fld(5, 2, attr_int("sorted", sorted_)))
    graph = (fld(1, 2, node) + s(2, "g") + fld(5, 2, k_tensor)
             + fld(11, 2, value_info("X", elem, [n, dim]))
             + fld(12, 2, value_info("V", elem, [n, k]))
             + fld(12, 2, value_info("I", 7, [n, k])))
    return fld(1, 0, 8) + fld(8, 2, fld(2, 0, 11)) + fld(7, 2, graph)


def run_case(dtype, dim, k, largest, sorted_, n=2):
    rng = np.random.default_rng(1234 + k)
    # integers in quarter steps are exact in fp16 and distinct for dim <= 16384
    base = (rng.permutation(dim)[None, :].repeat(n, 0) - dim // 2).astype(np.float32) / 4.0
    for r in range(1, n):
        base[r] = rng.permutation(base[r])
    x = base.astype(dtype)
    if dtype == "float16":
        # Quarter steps are exact only up to 512 in fp16, so larger values collide and the selected index set is
        # legitimately ambiguous at the K boundary. Distinct finite bit patterns give a total order.
        x = np.stack([rng.permutation(0x7bff)[:dim].astype(np.uint16).view(np.float16)
                      * np.where(rng.random(dim) < 0.5, -1, 1).astype(np.float16) for _ in range(n)])
    so = ort.SessionOptions()
    sess = ort.InferenceSession(topk_model(dtype, n, dim, k, largest, sorted_),
                                so, providers=["ROCMExecutionProvider"])
    if sess.get_providers()[0] != "ROCMExecutionProvider":
        raise RuntimeError("ROCm EP not active: %s" % sess.get_providers())
    t0 = time.time()
    v, i = sess.run(None, {"X": x})
    dt = time.time() - t0
    errs = []
    for r in range(n):
        ref_idx = np.argsort(-x[r].astype(np.float64) if largest else x[r].astype(np.float64),
                             kind="stable")[:k]
        ref_vals = x[r][ref_idx]
        if len(set(i[r].tolist())) != k or i[r].min() < 0 or i[r].max() >= dim:
            errs.append("row %d: indices not unique/in range" % r)
            continue
        if not np.array_equal(x[r][i[r]], v[r]):
            errs.append("row %d: values do not match x[indices]" % r)
        if sorted_:
            if not np.array_equal(v[r], ref_vals):
                errs.append("row %d: sorted values differ from reference" % r)
        else:
            if not np.array_equal(np.sort(v[r]), np.sort(ref_vals)):
                errs.append("row %d: unsorted value set differs from reference" % r)
            if set(i[r].tolist()) != set(ref_idx.tolist()):
                errs.append("row %d: index set differs from reference" % r)
    return errs, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-below", action="store_true",
                    help="also run K at or just under the bound (SafeSmallKTopK, slow)")
    args = ap.parse_args()

    bounds = {"float32": 7808, "float16": 7872}
    cases = []
    for dtype, b in bounds.items():
        above = [b + 1, b + 64, 12000]
        below = [b - 1, b]
        ks = above + (below if args.include_below else [])
        for k in ks:
            for largest in (1, 0):
                cases.append((dtype, max(k + 64, 12288), k, largest, 0))
        # sorted=1 is unaffected by the bound; K<=4096 is SafeSmallKTopK, K>4096 cub sort
        for k in (4096, 4097):
            cases.append((dtype, 12288, k, 1, 1))
        cases.append((dtype, 12288, 256, 0, 1))
        cases.append((dtype, 12288, 256, 0, 0))

    failed = 0
    for dtype, dim, k, largest, sorted_ in cases:
        errs, dt = run_case(dtype, dim, k, largest, sorted_)
        status = "ok  " if not errs else "FAIL"
        print("%s %-8s dim=%-6d K=%-6d largest=%d sorted=%d  %.2fs %s"
              % (status, dtype, dim, k, largest, sorted_, dt, "; ".join(errs)), flush=True)
        failed += bool(errs)
    print("%d/%d cases failed" % (failed, len(cases)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
