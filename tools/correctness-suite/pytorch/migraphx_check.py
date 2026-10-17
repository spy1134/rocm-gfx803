#!/usr/bin/env python3
"""MIGraphX gpu-target checks against float64/int64 numpy references.

  reduce_sum over int32 and uint32 tensors with a block-sized, a wave-sized and a sub-wave reduction axis. The
  integer DPP reduction has to assemble for gfx8; where it does not, compilation fails with "not a valid operand".
  dot in fp16 with a long reduction axis. The accumulation has to be fp32: an fp16 accumulator loses about
  1e-2 relative error at K=4096, an fp32 one stays within the fp16 rounding of the output (about 5e-4).

TARGET=ref runs the same programs on MIGraphX's reference target, which needs no GPU and checks the script itself.
"""
import os
import sys

import migraphx
import numpy as np

target = os.environ.get("TARGET", "gpu")
rng = np.random.default_rng(3)
bad = 0


def run(build, feeds):
    p = migraphx.program()
    mm = p.get_main_module()
    out = build(mm)
    mm.add_return([out])
    p.compile(migraphx.get_target(target))
    res = p.run({k: migraphx.argument(v) for k, v in feeds.items()})
    return np.array(res[0])


for dtype, mtype in ((np.int32, "int32_type"), (np.uint32, "uint32_type")):
    for rows, n in ((7, 16), (5, 64), (3, 256), (2, 1000), (4, 4096)):
        x = rng.integers(0, 50, size=(rows, n)).astype(dtype)
        name = f"reduce_sum {np.dtype(dtype).name} [{rows},{n}] axis 1"

        def build(mm):
            xs = mm.add_parameter("x", migraphx.shape(type=mtype, lens=[rows, n]))
            return mm.add_instruction(migraphx.op("reduce_sum", axes=[1]), [xs])

        try:
            got = run(build, {"x": x}).reshape(rows)
        except Exception as e:  # noqa: BLE001 - the failure text is the result
            print(f"FAIL {name}: {str(e).splitlines()[0][:110]}")
            bad += 1
            continue
        ok = np.array_equal(got.astype(np.int64), x.astype(np.int64).sum(axis=1))
        bad += not ok
        print(("ok   " if ok else "WRONG"), name)

for m, k, n in ((64, 4096, 64), (33, 4096, 17), (16, 2048, 128)):
    a = rng.standard_normal((m, k)).astype(np.float16)
    b = rng.standard_normal((k, n)).astype(np.float16)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    name = f"dot fp16 [{m},{k}]x[{k},{n}]"

    def build(mm):
        sa = mm.add_parameter("a", migraphx.shape(type="half_type", lens=[m, k]))
        sb = mm.add_parameter("b", migraphx.shape(type="half_type", lens=[k, n]))
        return mm.add_instruction(migraphx.op("dot"), [sa, sb])

    try:
        got = run(build, {"a": a, "b": b}).reshape(m, n).astype(np.float64)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}: {str(e).splitlines()[0][:110]}")
        bad += 1
        continue
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    ok = rel < 2e-3
    bad += not ok
    print(("ok   " if ok else "WRONG"), name, f"rel_l2={rel:.2e}")

print(f"RESULT migraphx_check target={target} bad={bad}")
sys.exit(1 if bad else 0)
