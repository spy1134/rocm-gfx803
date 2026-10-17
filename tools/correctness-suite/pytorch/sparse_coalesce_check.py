"""Sparse COO coalesce() against a float64 CPU result, over dense widths around the
160- and 256-feature tile boundaries of coalesceValuesKernel.

Run on the GPU under test:  python3 -u sparse_coalesce_check.py
`--device cpu` runs the same checks on the CPU (self test of this script).
Exit status 0 only if every case matches.
"""
import argparse
import sys

import torch


def case(dev, width, dtype, rows=4, nnz=12, seed=0):
    g = torch.Generator().manual_seed(seed + width)
    idx = torch.randint(0, rows, (1, nnz), generator=g)        # duplicates guaranteed (nnz > rows)
    if dtype == torch.bool:
        vals = torch.rand(nnz, width, generator=g) > 0.7
    else:
        vals = torch.randn(nnz, width, generator=g).to(dtype)
    ref_dtype = torch.bool if dtype == torch.bool else torch.float64
    ref = torch.sparse_coo_tensor(idx, vals.to(ref_dtype), (rows, width)).coalesce()
    got = torch.sparse_coo_tensor(idx.to(dev), vals.to(dev), (rows, width)).coalesce()
    if not torch.equal(got.indices().cpu(), ref.indices()):
        return False, "indices differ"
    gv, rv = got.values().cpu(), ref.values()
    if dtype == torch.bool:
        return bool(torch.equal(gv, rv)), "bool values"
    # The reference sums the fp16/fp32 inputs in float64; the device accumulates in fp32.
    tol = 2e-3 if dtype == torch.float16 else 1e-5
    err = (gv.double() - rv).abs().max().item()
    wrong = int(((gv.double() - rv).abs() > tol * (1 + rv.abs())).sum())
    return wrong == 0, f"max err {err:.2e}, {wrong} wrong of {rv.numel()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("no accelerator; nothing to test")
        return 0
    bad = 0
    for dtype in (torch.float32, torch.float16, torch.bool):
        for width in (1, 31, 32, 64, 128, 159, 160, 161, 192, 255, 256, 257, 300, 512, 1024):
            ok, msg = case(args.device, width, dtype)
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {str(dtype):14s} width={width:<5d} {msg}")
    print(f"{bad} failing case(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
