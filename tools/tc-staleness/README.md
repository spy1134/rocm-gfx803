# tools/tc-staleness

These are hardware probes for the gfx803 cross-dispatch coherence bug. On gfx7 and
gfx8 the command processor ignores the AQL acquire and release fence scopes, which
are ROCm's only expression of cross-dispatch coherence, so cache lines that a
buffer instruction fetched through a descriptor with word3 bit 27 clear stay valid
in the per-CU vector L1 and scalar cache from one kernel dispatch to the next. A
kernel that reads a page a copy kernel has just refilled, at a virtual address the
allocator recycled, can be served the page's old contents, silently.
`patches/clr/clr-gfx8-tc-invalidate-6.4.4.patch` rebuilds `libamdhip64`
with a PM4 `ACQUIRE_MEM` (vector L1 and scalar cache) ahead of each kernel
dispatch; `CLR_GFX8_TC_INVALIDATE=0` switches it off in the same binary. The patch
header holds the mechanism, the evidence, the cost and which cache bits matter.

How to read the results. `loadpaths.hip` shows the mechanism directly: one row per
load path (flat, scalar, buffer loads with bit 27 clear and set) and one column per
kernel-entry invalidate, counting stale reads of a recycled buffer. With the fix
off, only the buffer loads with bit 27 clear read stale data. `probe_rocblas.py` is
the probe that fails without the fix through a real library: about 500 small rocBLAS calls that each allocate, copy in,
compute, copy out and free, stopping at case 225 when the L1 is stale (and
passing all 511 with `NOFREE=1`, which proves the failure needs a recycled
address). The PyTorch-level probes below are clean on ROCm 6.4.4 with the fix off
as well, because PyTorch's kernels load through flat instructions, so a clean
result from them says nothing about the fix; they stay as
regression checks and for stacks where the bug shows up through PyTorch. The
reader direction needs a recycled virtual address, survives `synchronize()`,
`AMD_SERIALIZE_KERNEL=3`, SYSTEM-scope fences and GPUVM TLB invalidation, and is
not an out-of-bounds read (`PAD_BYTES`/`PAD_FILL` show that). The copy-engine
direction (the GPU's own answer correct, the host copy of it wrong) does not
reproduce on 6.4.4, where ROCr uses no SDMA on gfx8 by default; `conbrace3.py`
compares an on-device error against a post-copy error to tell the two apart.

## Running

You need a gfx803 card and a HIP stack, and nothing else.

```sh
hipcc --offload-arch=gfx803 -O2 loadpaths.hip -o loadpaths
CLR_GFX8_TC_INVALIDATE=0 ./loadpaths              # stale reads in the bit-27-clear rows only
python3 -u probe_rocblas.py --max-n 64 --reps 6   # stops at case 225 with CLR_GFX8_TC_INVALIDATE=0
NOFREE=1 python3 -u probe_rocblas.py              # control: hipFree is a no-op, passes
python3 probe.py --trials 40                      # default shape 2x128x64x128
python3 probe.py --shape 4 256 128                # other bmm shapes
```

`loadpaths`, `probe_rocblas.py` and `probe.py` exit 0 when clean and 1 when stale reads are present, so
they drop into a script:

```sh
CLR_GFX8_TC_INVALIDATE=0 python3 -u probe_rocblas.py && echo "the probe does not detect the bug here"
python3 -u probe_rocblas.py || echo "regression: stale reads back"
```

## Harnesses here

| file | what it covers |
|---|---|
| `loadpaths.hip` | stale reads of a recycled 4 KiB buffer per load path (flat, `s_load`, `buffer_load` and `s_buffer_load` with descriptor bit 27 clear and set) and per kernel-entry invalidate (`buffer_wbinvl1_vol`, `buffer_wbinvl1`, `s_dcache_inv`) |
| `probe_rocblas.py` | rocBLAS level-1/2/3 calls against float64, each case allocating and freeing its buffers; the probe that isolates the bug. `_hip.py` is its ctypes layer |
| `probe.py` | bmm, checked three ways (direct / frozen GPU->GPU copy / re-read) + repeat determinism |
| `soak.py` | 527 checks across bmm, conv+silu+groupnorm, linear+layernorm+softmax, `scaled_dot_product_attention`, reductions (sum/amax/std/cumsum/sort/median/logsumexp) and gather/scatter_add, in fp16 and fp32 |
| `train.py` | real training loop (4-conv + GroupNorm + SiLU + linear + LayerNorm, 60 Adam steps): loss vs an identical CPU run, bit-stability of the same seed, frozen-copy check on the gradients |
| `conbrace3.py` | suite-shaped conv loop that classifies each failure by site: device-side inputs, error recomputed on the GPU, error after the host copy. `copy_only: N` with `bad_gpu: 0` means the conv was right and the copy was wrong |
| `multistream2.py` | cross-stream producer to host copy. `SYNC=1` puts a full device sync between the producer and the copy; that measures clean, but it is a serialisation, not a fix, and nothing in a real pipeline can afford it |
| `ms3.py` | `multistream2`'s shape with each anomaly classified by site, so a failure names its own cause instead of just counting |
| `ms4.py` | the same handoff with a pinned host destination, and with two producer streams fanning into one consumer |
| `graphprobe.py` | a captured `CUDAGraph` that reads what a copy engine, or a kernel on another queue, wrote. This is the probe for the one dispatch path the patch leaves uncovered |
| `xprobe2.py` | the cross-stream reader direction, with no copy engine involved |
| `pinnedhost.py` | a kernel writes pinned host memory and the CPU reads it after a stream sync |
| `wavesize_proof.hip` | rocSOLVER's wave-partial reduction pattern at `WarpSize` 32 and 64, see `patches/rocsolver/`. Build it with `hipcc --offload-arch=gfx803` |

Run each probe twice, once with `CLR_GFX8_TC_INVALIDATE=0` as the control. Only a probe
that fails with the knob off can show the fix at work; on ROCm 6.4.4 those are
`loadpaths.hip` and `probe_rocblas.py`. The cross-stream ones (`multistream2`, `ms3`, `ms4`) also take
`CLR_GFX8_TC_RECORD_FENCE=0`, which turns off only the event-record site. The end-to-end
gate is `tools/correctness-suite/torch_op_suite.py`.

## Expected results (gfx803, ROCm 6.4.4, one binary, knob toggled)

```
loadpaths      (off) -> stale reads of 1,000, by kernel entry (none, wbinvl1_vol, wbinvl1, dcache_inv):
                        flat_load 0 0 0 0; buffer_load bit 27 clear ~995 ~995 0 ~995;
                        buffer_load bit 27 set 0 0 0 0; s_load 0 0 0 0;
                        s_buffer_load bit 27 clear 989 989 989 0; s_buffer_load bit 27 set 0 0 0 0
loadpaths      (on)  -> 0 in every cell
probe_rocblas  (off) -> WRONG at case 225 (sgemm_strided_batched m=n=k=8 b=4, rel err 0.94)
probe_rocblas  (on)  -> 511/511
probe_rocblas  NOFREE=1, off -> 511/511
probe.py, soak.py, conbrace3.py, ms3.py, graphprobe.py
                     -> clean with the knob on and with it off (631 soak checks,
                        40 probe trials, 240 conbrace3/ms3 checks, 300 graph checks)
```

The correctness suite and the 20-step training run pass with the knob on. The
probe results above were taken with the packet word 0x08400000.

