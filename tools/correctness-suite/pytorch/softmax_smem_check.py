"""softmax / log_softmax forward and backward against float64, at the last-dim sizes
where the block-reduction shared-memory buffer can overrun the allocation (large rows
through cunn_SoftMaxForwardSmem) and over the persistent-softmax range (rows <= 2048).

Run on the GPU under test:  python3 -u softmax_smem_check.py
`--device cpu` runs the same checks on the CPU (self test of this script).
Exit status 0 only if every case is within tolerance and every row sums to 1.
"""
import argparse
import sys

import torch
import torch.nn.functional as F

PERSISTENT = [33, 63, 64, 65, 127, 129, 1000, 1024, 2047, 2048]
LARGE_FP32 = [9316, 9320, 9324, 9328, 9332, 9444, 9448]      # 9332 is a control
LARGE_FP16 = [9416, 9424, 9432, 9440, 9672]


def one(dev, d, dtype, log):
    rows = 8 if d < 4096 else 3
    g = torch.Generator().manual_seed(d)
    x = (torch.randn(rows, d, generator=g) * 3).to(dtype)
    go = torch.randn(rows, d, generator=g).to(dtype)
    fn = F.log_softmax if log else F.softmax

    xg = x.to(dev).clone().requires_grad_()
    yg = fn(xg, dim=-1)
    yg.backward(go.to(dev))

    xr = x.double().clone().requires_grad_()
    yr = fn(xr, dim=-1)
    yr.backward(go.double())

    # fp16 outputs are rounded to 11 bits, so the tolerance scales with the output magnitude.
    tol = 3e-3 if dtype == torch.float16 else 2e-5
    ey = (yg.detach().cpu().double() - yr.detach()).abs().max().item() / (1 + yr.detach().abs().max().item())
    eg = (xg.grad.cpu().double() - xr.grad).abs().max().item() / (1 + xr.grad.abs().max().item())
    rowsum = yg.detach().cpu().double().exp().sum(-1) if log else yg.detach().cpu().double().sum(-1)
    es = (rowsum - 1).abs().max().item()
    ok = ey < tol and eg < tol and es < 10 * tol
    return ok, f"fwd {ey:.1e} bwd {eg:.1e} rowsum-1 {es:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("no accelerator; nothing to test")
        return 0
    bad = 0
    plan = [(torch.float32, d) for d in PERSISTENT + LARGE_FP32]
    plan += [(torch.float16, d) for d in PERSISTENT + LARGE_FP16]
    for dtype, d in plan:
        for log in (False, True):
            ok, msg = one(args.device, d, dtype, log)
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {'log_softmax' if log else 'softmax    '} "
                  f"{str(dtype):14s} dim={d:<5d} {msg}")
    print(f"{bad} failing case(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
