"""opinfo_fp16.py [OP...]: every OpInfo op that runs in float16 on the GPU, over its reference inputs, against the same
inputs upcast to float32 on the CPU (the result cast back to float16), atol and rtol 1e-2; one RESULT line per op (see
opinfo_compare_cpu.sh --fp16). PyTorch's own test_compare_cpu checks one dtype per op, float32 for most."""
import sys, time, torch
from torch.testing._internal.common_methods_invocations import op_db
torch.cuda.set_per_process_memory_fraction(0.6)
names = set(sys.argv[1:])
tree_map = torch.utils._pytree.tree_map
def to_cpu32(x):
    if isinstance(x, torch.Tensor):
        x = x.to("cpu")
        return x.float() if x.dtype == torch.float16 else x
    return x
def to16(x):
    return x.half() if isinstance(x, torch.Tensor) and x.dtype == torch.float32 else x
for op in op_db:
    if names and op.formatted_name not in names:
        continue
    if torch.float16 not in op.supported_dtypes("cuda") or torch.float32 not in op.supported_dtypes("cpu"):
        print(f"RESULT {op.formatted_name} skip 0.0s no float16 on the GPU", flush=True)
        continue
    t0, status, detail = time.time(), "ok", ""
    try:
        for i, s in enumerate(op.reference_inputs("cuda", torch.float16)):
            c = s.transform(to_cpu32)
            g = tree_map(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x,
                         s.output_process_fn_grad(op(s.input, *s.args, **s.kwargs)))
            r = tree_map(to16, c.output_process_fn_grad(op(c.input, *c.args, **c.kwargs)))
            try:
                torch.testing.assert_close(g, r, atol=1e-2, rtol=1e-2, equal_nan=True, check_dtype=False, check_device=False)
            except AssertionError as e:
                status, detail = "FAIL", f"sample {i}: " + str(e).replace("\n", " | ")[:400]
                break
    except Exception as e:
        status, detail = "ERROR", (type(e).__name__ + ": " + str(e)).replace("\n", " | ")[:300]
    print(f"RESULT {op.formatted_name} {status} {time.time() - t0:.1f}s {detail}", flush=True)
print("GROUPDONE", flush=True)
