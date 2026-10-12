# Optimization targets for gfx803

Performance work on the `rocm6.4.4/` line: Stable Diffusion 1.5 in ComfyUI,
SD 1.5 LoRA training with sd-scripts, and llama.cpp through Ollama, on Polaris
(gfx803). This file says where the time goes today, which targets are worth
taking on next and in what order, what each is expected to gain and why, how
to measure and verify the work, and what has already been done or ruled out.
All measurements are from one RX 590.

Correctness comes first here. A faster kernel counts only once its results are
bit-identical to the kernel it replaces, or match a CPU float64 reference, and
the full checks in "Verifying a change" pass. Correctness problems and their
fixes are documented in [`FINDINGS.md`](FINDINGS.md).

## The card

| | RX 590 (Polaris 30) | What it means for kernels |
| --- | --- | --- |
| fp32 | 36 CUs, 2304 lanes, ~1.55 GHz: ~7.1 TFLOPS (FMA) | The card's real strength. Measure GEMM-shaped work against it. |
| fp16 | same rate as fp32 | No packed fp16 math (`v_pk_*`): fp16 halves memory traffic and storage, not arithmetic time. |
| Integer dot products | none | No `v_dot4`/`dp4a`. Quantized dot products are ALU sequences; 24-bit integer multiplies (`v_mul_i32_i24`) are full rate, and SDWA byte/word operand selects are free. |
| Matrix cores | none | Every GEMM is plain FMA from registers. |
| Memory | 256 GB/s GDDR5 nominal, ~225 GB/s reached by streaming kernels | Single-token LLM decode, norms and elementwise ops are bound here. 28 FLOP per byte separates compute-bound from memory-bound work. |
| DRAM channels | 16 | Power-of-two leading dimensions can put a wavefront's accesses on one channel (partition camping; it cost the shim 20x). |
| Per CU | 64 KiB LDS; 4 SIMDs with 64 KiB of VGPRs each (256 VGPRs per lane); 64-lane wavefronts | Up to 128 VGPRs a SIMD holds 2+ waves, above it one; above 32 KiB of LDS per block a CU holds one block. |
| L2 | 2 MiB | |
| VRAM | 8 GB, shared with the desktop if the card drives it | Desktop framebuffers are invisible to ROCm's free-memory figure. |
| Host link | PCIe 3.0 x16, ~13 GB/s | Bounds anything that streams weights from system RAM (MoE experts, offloaded layers). |

The rest of the family (RX 470/480/570/580: 32-36 CUs at lower clocks, same
256-bit bus) has the same balance, so the conclusions here should carry over,
scaled by clock. Polaris 11/12 (RX 460/550/560: 8-16 CUs, 64- or 128-bit
bus) is untested.

What follows from this: the card does dense fp32 multiply-add well, as long as
the data is reused from registers and LDS. Almost everything expensive in
these workloads is matrix multiplication: linear layers, convolutions (as
Winograd plus a batched GEMM), attention (two GEMMs around a softmax) and LLM
prompt processing. Making all of it run like the shim's best GEMM kernel does
today is the main lever. Today only one GEMM layout gets there.

## Where the time goes

### SD 1.5 inference

One denoising step, fp32, 512 px, classifier-free guidance (batch 2), ComfyUI
with `--use-split-cross-attention` and the gfx803 node: 526 ms (644 without
the node). Time per module class, from CUDA events around every UNet module's
forward, measured at 566 ms per step, before BLAS-5 took the linear layers'
TN GEMMs from ~2.7 to 3.6-4.4 TFLOPS:

| What | Time | How it runs | Efficiency |
| --- | --- | --- | --- |
| Linear layers (attention projections, feed-forward, proj in/out) | 172 ms (30%) | shim, TN layout (all 184 linears per step) | ~2.7 TFLOPS |
| Convolutions | 182 ms (32%) | MIOpen `ConvHipWinogradF2x3`: input/output transforms in HIP kernels around a strided-batched GEMM in the shim (NN, 438 GFLOP per step, ~97 ms at ~4.5 TFLOPS) | ~85 ms is outside the GEMM |
| Attention, excluding its projections | ~150 ms (~26%) | node's fused kernel; 64x64 self-attention is 26 ms per call, 5 calls per step | ~1.65 TFLOPS |
| GroupNorm, LayerNorm, SiLU, the rest | ~13 ms (2%) | | |

The attention row is an estimate: CrossAttention modules measured 220 ms
including their own projections, and the projections' share was taken from
their FLOP counts.

GEMM FLOPs per step by layout (shim call log, after a warm-up pass), without
the node:

| Call | Layout | GFLOP | Share | Mostly |
| --- | --- | --- | --- | --- |
| `sgemm` | TN | 467 | 40% | `torch.nn.Linear` |
| strided-batched `gemm_ex` | NN | 438 | 37% | the Winograd solver's GEMM |
| strided-batched `sgemm` | TN, NN | 126 + 126 | 22% | split attention's two bmms (the node replaces these) |
| `gemm_ex` | NN | 12 | 1% | GEMM-based convolutions |

Layouts are rocBLAS's column-major view: TN means op(A) = transposed,
op(B) = not transposed.

### SD 1.5 LoRA training

A 20-step fp32 run (34-image set, 512 px buckets, gradient checkpointing,
batch 1, gradient accumulation 4) takes about 3.4 s per step at steady state.
GEMM FLOPs by layout for one UNet forward and backward with frozen weights,
which is what LoRA training's base model does, at 384 px (512 px does not fit
in 8 GB without gradient checkpointing):

| Layout | Share |
| --- | --- |
| NN | 50% |
| TN | 24% (forward linears, attention) |
| NT | 26% (backward of convolutions, linears and attention) |

Training's attention runs as plain bmm with a stored score matrix, since the
node is forward-only.

### llama.cpp decode (token generation)

Qwen3 8B Q4_K_M with a 5.2k-token context generates 26.6 tok/s, 37.6 ms per
token. Reading the ~5 GB of weights at the measured ~180 GB/s of the K-quant
decode kernels accounts for ~28 ms, and the q8_0 KV cache for ~2 ms. The
remaining ~7 ms is unexplained. It is not CPU launch overhead that HIP graphs
would remove: llama.cpp captures and replays graphs on gfx803 by default, and
turning them off (`GGML_CUDA_DISABLE_GRAPHS=1`) leaves generation unchanged
(30.6 against 30.5 tokens/s). The K-quant matmuls
themselves are at ~80% of the reachable bandwidth (q4_K 4096x14336: 188 us,
~176 GB/s; q8_0 and f16 reach ~225).

### llama.cpp prompt processing (prefill)

Batches of 128 tokens or more run as fp32 GEMMs on the shim (LLM-4); smaller
ones, and MoE expert matmuls, run through mmq, which emulates int8 dot
products at about 2.4 effective TOPS. Qwen3 8B reads a 5.2k-token prompt in
Ollama at ~215 tok/s and Qwen2.5-Coder 7B at ~136. A 12,288-token prompt, the
size of a coding agent's first request, takes 71 s (173 tok/s; unmodified
llama.cpp took 18 minutes, at 11.4 tok/s). MoE prompts: LLM-6.

## Status

Open targets in recommended order, then the rest.

| Target | Area | Expected gain | Status |
| --- | --- | --- | --- |
| LLM-6 | MoE prompts: expert copies, MUL_MAT_ID, dense parts | expert copies now overlap compute (patch 108: pp2048 295 -> 367 tok/s); MUL_MAT_ID next | Open, first |
| LLM-8 | MoE generation with experts in system RAM | see LLM-8; 42 ms per token on Qwen3.6 35B-A3B, about 17 ms of it RAM-bound | Open, measured |
| PT-2 | PyTorch kernels that mix the host's 64-lane warp size with the device's 32 | correctness: wrong results or a GPU reset | Open, audit (no GPU risk) |
| SD-6 | Register-tiled fused attention (and llama.cpp's fp32 FA tile kernel) | SD inference ~12% per step; deep-context prompts | Open |
| SD-7 | Fused Winograd F(2x2, 3x3) | SD 30-60 ms per step (5-10%), training too | Open |
| LLM-5 | The unexplained ~7 ms per decoded token | unknown, up to ~19% generation | Open, measure first |
| SD-2 | Winograd F(4x4, 3x3) | unmeasured; only after SD-7 | Open |
| SD-1 | Winograd weight gradients | full fine-tuning and LoCon only | Open |
| SD-8 | Fused attention backward | training attention time and VRAM | Open, large |
| LLM-3 | mmvq vs mmq for 9-16 tokens | speculative/parallel decoding | Open |
| LLM-1b | SDWA dot product losses (2-token decode) | 1-11% on 2-token decode | Open, minor |
| LLM-4 | llama.cpp prompts through dequantize + fp32 SGEMM (patch 106) | Qwen3 8B prompt 149 -> 215 tok/s in Ollama | Done |
| LLM-7 | DeltaNet conv-input concat (patch 107) | Qwen3.6 35B-A3B pp2048 244 -> 270 tok/s | Done |
| BLAS-6 | Shim: GEMMs with 1-8 columns (decode-shaped), fp16 and fp32 | Chatterbox TTS token model 88 -> 25.7 ms per token | Done |
| BLAS-5 | Shim: TN, NT and TT layouts | SD step 557 -> 526 ms, training 3.63 -> 3.37 s/step | Done |
| BLAS-1, BLAS-2 | Shim: pipelined 64x64 and 32x32 tiles | large NN GEMMs 2.8-3.3 -> 3.9-4.8 TFLOPS | Done |
| BLAS-4 | Shim: fp16 GEMMs in every layout | slower than rocBLAS | Evaluated, not adopted |
| SD-3, SD-4, SD-5 | ComfyUI node: fused attention, GroupNorm, GEGLU | SD step 697 -> 567 ms | Done, optional install |
| PT-1 | PyTorch layer_norm without the group_norm route | correctness (and ~1%) | Done |
| LLM-1 | llama.cpp SDWA dot product in decode kernels (patch 104) | Q4_K_M generation +6-7% | Done |
| LLM-2 | llama.cpp q5_K decode (patch 105) | q5_K decode ~4% | Done |

## Open targets

### LLM-6: MoE prompt processing

**Why.** The MoE models used most here (Qwen3.6 35B-A3B: 256 experts, top 8,
18.7 GiB of expert weights; Qwen3-Coder-Next: 512 experts, top 10, 43.6 GiB)
keep most experts in system RAM. For a batch of 32 tokens or more, llama.cpp
copies each layer's experts to the GPU (op offload) and runs them there with
mmq's MUL_MAT_ID kernel. Patch 106 does not apply to the experts: each expert
sees about batch x top-k / experts tokens (64 at a 2048-token batch), below
the 128 where fp32 cuBLAS beats mmq.

**Where the time goes.** Qwen3.6 35B-A3B Q4_K_M, all experts in RAM
(`-ncmoe 99`), 2048-token batch, per 2048 tokens (7.6 s at 270 tok/s), from a
dev-only timer that synchronizes around every CUDA node:

| Part | Time | Notes |
| --- | --- | --- |
| Expert copies over PCIe | ~1.8 s | 20 GB at 11.3 GB/s from pinned memory, once per batch |
| MUL_MAT_ID (experts) | 2.2 s | 1.4-2.0 TFLOPS; q6_K slower than q4_K |
| Dense MUL_MAT | ~1.5 s | after patch 106 and the shim's alignment rule |
| Flash attention | 0.2 s | |
| Everything else | ~1.9 s | CPU-side scheduling and synchronization, not broken down yet |

Copy time is per batch, so a larger batch amortizes it: 174 / 224 / 270
tok/s at batches of 512 / 1024 / 2048.

**Options.**
- Done: patch 108 copies the next MoE layer's experts into a second device
  slot on a second stream while the current layer computes, with
  `HSA_ENABLE_SDMA=1` so the copy runs on the DMA engine (FINDINGS section
  9). pp2048 at 512 / 1024 / 2048-token batches: 193 / 250 / 295 -> 243 /
  321 / 367 tok/s, against a 376 tok/s bound with no copies at all. At
  512-token batches the copy is still longer than the compute it hides
  behind (12 GB/s over PCIe).
- MUL_MAT_ID tile choice for gfx803: measured, not a lever. Forcing every
  tile width from 8 to 64 for the expert matmuls (dev-only override) left
  pp2048 within noise: 171-176 tok/s at 1024-token batches for every width,
  238-257 at 2048.
- The PCIe link speed matters as much as any kernel. amdgpu's PCIe DPM can
  stay at its low level (2.5 GT/s x8) under load: `pp_dpm_pcie` then shows
  level 0 throughout a copy, a 512 MiB host-to-device copy runs at 3.4
  instead of 12.3 GB/s, and the same build's pp2048 (2048-token batches)
  falls from 367 to 247 tok/s. Generation barely changes. Check
  `current_link_speed` / `pp_dpm_pcie` under load before comparing prompt
  numbers; switching transparent huge pages off for the process changed
  nothing.
- Done for the Qwen3.6 35B text entries: `num_batch` 2048 with patch 108
  (2k-token prompt 142 -> 198 tok/s, about 2% slower generation). The vision
  entry stays at 1024 (short image prompts gained nothing; generation 13.4 ->
  12.2). Qwen3-Coder-Next is unmeasured: it pins 51.5 GiB of RAM and does not
  fit beside Chatterbox, Open WebUI and a desktop on 64 GB.

**Verify.** test-backend-ops `MUL_MAT_ID` (the dev tree adds perf cases at
these models' expert shapes); llama-perplexity KL divergence against the CPU
backend; Ollama greedy output on qwen3.6:35b.

### LLM-8: MoE generation with experts in system RAM

**Where the time goes.** Qwen3.6 35B-A3B Q4_K_M, all experts in host memory
(`-ncmoe 99`), llama-bench tg: 23.7 tok/s, 42 ms per token.

| Part | Per token | How it was measured |
| --- | --- | --- |
| CPU expert matmuls | ~17 ms | GGML_SCHEDTIME split timer; 0.56 GB of Q4_K expert weights per token is ~33 GB/s, the practical limit of dual-channel DDR4-2800 |
| GPU splits (attention, DeltaNet, router, shared expert, output head) | ~17 ms | split timer minus the hand-off share below; about 600 nodes per token |
| GPU-CPU hand-offs (two per layer, 80 per token) | ~8 ms | moving 4 layers' experts to the GPU (`-ncmoe 36`) saves 0.53 ms per layer against 0.42 ms of CPU expert time |

What does not change it: CPU threads (4: 21.6, 8: 23.7, 12: 23.0, 16: 18.6
tok/s), the thread pool's polling (`--poll` 0/50/100), HIP graphs (on: 23.4,
off: 24.0; they are captured once per split and replayed, 2523 replays to 71
captures), and `ROC_ACTIVE_WAIT_TIMEOUT` (100 to 10000 us).

**Open questions.** The GPU splits read about 1 GB of weights per token
(5 ms at 200 GB/s) plus about 600 small kernels, which test-backend-ops runs
at 6.7 us each back to back (ADD, 4096 floats); 17 ms is about twice that. A
kernel timeline would settle where the rest goes, but rocprofv3 on this
stack aborts in its own shutdown before writing the trace (see Timeline
below for the profiler that works). The per-node timer (GGML_OPTIME) charges fused groups to their
first node (sigmoid + mul shows as a 43 us SIGMOID), so its per-op numbers
are not kernel times.

**Timeline.** The legacy `rocprof --hip-trace` works on this stack when
nothing is preloaded (`unset LD_PRELOAD`, so the image's patched HSA runtime
loads); rocprofv3 aborts at exit even for a trivial PyTorch script, and
rocprofv2 has no gfx803 counter definitions. In a 1.5 s window of generation
(about 30 tokens, slowed to 13 tok/s by the tracing), the GPU runs kernels
33% of the time and sits in barrier packets (waiting for the CPU) 46%. Per
token, the host makes about 300 `hipStreamSynchronize` calls: 130 right after
a `hipMemcpyAsync` (90 of them device-to-device copies into pinned host
buffers, the GPU results the CPU expert split reads), 128 right after another
synchronize, and 41 after a graph launch (one per GPU split). The minimum is
about one wait per GPU-to-CPU hand-off (41). Kernel time per token is
dominated by `mul_mat_vec_q` (about 270 calls, 32 us mean under tracing).

**GPU kernels by shape.** With `GGML_CUDA_DISABLE_GRAPHS=1` every kernel goes
through `hipLaunchKernel`, whose trace record carries the template (quant
type) and grid, so each `mul_mat_vec_q` can be matched to its weight
(rows = grid x 2 for the two-row kernel). Per generated token:

| Weight | Calls | Each | Effective bandwidth |
| --- | --- | --- | --- |
| output head, 248320 x 2048 q6_K | 1 | 2.09 ms | ~200 GB/s |
| 8192 x 2048 | 43 | 63 us | 150-220 GB/s |
| 4096 x 2048 | 32 | 37 us | ~130 GB/s |
| 2048 x 512 (shared-expert down) and 2048 x 4096 | 85 | 22 us | the K=512 ones ~27 GB/s |
| 512 x 2048 (shared-expert gate, up) | 64 | 15 us | ~40 GB/s |
| router, 256 x 2048 fp32 (`mul_mat_vec_f`) | 43 | 21 us | ~100 GB/s |

The large weights run near the card's bandwidth; the shared expert's small
matrices do not: each workgroup has almost nothing to do, so a launch costs
15-22 us for under 1 MB. A small-matrix variant (more rows per workgroup
when rows x K is small) or one launch for gate and up would recover most of
an estimated 1.5-2 ms per token (3-5%); the fp32 router about 0.5 ms more.
`gated_delta_net` (30 per token, 29 us, ~4 MB of state each) is at about
140 GB/s.

**Levers.** Done: patch 109 copies a CPU split's GPU inputs with one wait
instead of one per tensor (tg128 23.8 -> 24.4 tok/s). The back-to-back
duplicate synchronizations are nearly free (an idle-stream synchronize
costs about 1 us); a small copy or kernel plus its wait costs 24-29 us.
Each layer whose experts stay in VRAM saves about 0.53 ms per token, so VRAM
taken by other programs costs generation speed directly (with Chatterbox
loaded, Ollama keeps 3.2 instead of 4.4 GiB of this model on the card).

### PT-2: audit PyTorch's warp-size mixing

In gfx8 device code `C10_WARP_SIZE` is 32 (`c10/macros/Macros.h` gives 64 to
gfx9 only), while the hardware runs 64-lane wavefronts and host code's
`at::cuda::warp_size()` returns 64. A kernel that is launched or sized by
one and computes with the other is wrong on gfx803 and correct on every
architecture upstream tests. Three cases are fixed: layer_norm (PT-1); the
binary prefix scan behind `torch.topk`
([`gfx803-topk-scan-warp-size.patch`](../rocm6.4.4/patches/pytorch/gfx803-topk-scan-warp-size.patch)),
which wrote top-k results to wrong slots and, through its assert, hung the GPU
until the driver reset it; and index_put with accumulate
([`gfx803-index-put-warp-size.patch`](../rocm6.4.4/patches/pytorch/gfx803-index-put-warp-size.patch)),
which dropped half of the repeated indices' sums. group_norm shows the
opposite dependency: its weight and bias gradients need the 32, so the macro
itself cannot change. FINDINGS section 5 describes all of them.

At commit d618db8, 21 files under `aten/` use `C10_WARP_SIZE`, and 14 of them
also call `at::cuda::warp_size()`:

- `aten/src/ATen/native/cuda/block_reduce.cuh`
- `aten/src/ATen/native/cuda/group_norm_kernel.cu`
- `aten/src/ATen/native/cuda/Sorting.cu`
- `aten/src/ATen/native/cuda/Repeat.cu`
- `aten/src/ATen/native/cuda/PersistentSoftmax.cuh`
- `aten/src/ATen/native/cuda/TensorTopK.cu`
- `aten/src/ATen/native/cuda/SoftMax.cu`
- `aten/src/ATen/native/cuda/Embedding.cu`
- `aten/src/ATen/native/cuda/DepthwiseConv2d.cu`
- `aten/src/ATen/native/cuda/Indexing.cu`
- `aten/src/ATen/native/cuda/layer_norm_kernel.cu`
- `aten/src/ATen/native/cuda/Normalization.cuh`
- `aten/src/ATen/native/cuda/EmbeddingBag.cu`
- `aten/src/ATen/native/cuda/DepthwiseConv3d.cu`

`layer_norm_kernel.cu` and the two index_put launchers in `Indexing.cu` are
done; `TensorTopK.cu`'s single-block path is done through `ScanUtils.cuh`,
and its multi-block path (`mbtopk`) is not yet read.

Mixing the two is not a bug by itself: a kernel whose shuffles use offsets
below 32 and whose shared memory is sized from `blockDim` treats each half of
a 64-lane wavefront as its own warp and stays correct (layer_norm's kernels
are like that once launched consistently). The bug is a disagreement inside
one kernel: a 64-lane ballot or lane-0 store counted with 32-lane warp
numbers (the top-k scan), a launch or shared-memory size from one value and
the kernel's indexing from the other (layer_norm), or a fixed block shape
that assumes 32 lanes per warp. For each file, list every kernel, the warp
size its launch and shared memory use, and the warp size its device code
assumes (`C10_WARP_SIZE`, `warpSize`, `__lane_id()`, ballots, shuffle widths),
and mark the disagreements. Softmax and group_norm already match the CPU in
the image's op suite, which narrows their part to the paths the suite does
not reach.

The audit needs no GPU. Each suspect it finds then gets a GPU-against-CPU
test for the affected op, run headless with `AMD_SERIALIZE_KERNEL=3` so that a
trapping assert names its kernel and costs one reset. PyTorch's own OpInfo
tests (about 600 ops, against the suite's 118) run the same way, one op per
process, would find the cases the audit misses.

### SD-6: register-tiled fused attention

**Why.** The node's fused attention removes the 2.3 GB score matrix and
halves self-attention time, but runs at ~1.65 TFLOPS: 64x64 self-attention
(batch 2, 8 heads, 4096 tokens, head size 40) is 43 GFLOP in 26 ms, and its
five calls are ~130 ms of a 566 ms step. Attention is two GEMMs around a
softmax, which is the card's best kind of work.

**Where.** `tools/gfx803_fused_attention.hip` (kernels for head sizes 40/64
and a 2-wave variant for 80/160) and `tools/gfx803_attention.py` (dispatch).

**Approach.** Apply the shim's lessons: a larger output tile per thread
(registers hold more of Q and the running output), K and V tiles staged
through LDS with the next tile's loads issued before the current tile's math,
and float4 global loads. Head size 40 is not a multiple of 16; padding to 48
or 64 in LDS may beat handling the tail. Keep the online-softmax arithmetic
in fp32.

**Expected gain.** ~3.5 TFLOPS would take 64x64 self-attention from 26 to
~12 ms per call: about -70 ms per step (~12%).

**Shared with llama.cpp.** Patch 101's fp32 flash-attention tile kernel
(`fattn-tile-kernel.cuh`, the `fattn_tile_fp32` namespace) is the same kind
of kernel. Prompts 8k tokens deep still read at 127 tok/s against 163 at an
empty context (Qwen3 8B, q8_0 KV), so the same design may help there.

**Verify.** CPU float64 reference over head sizes 40/64/80/160, odd N and M,
3-D and non-contiguous inputs, large score magnitudes (online softmax
overflow), fp16 through the hook; then a full ComfyUI txt2img against a CPU
run.

### SD-7: fused Winograd F(2x2, 3x3)

**Why.** Convolutions are 182 ms (32%) of an SD step, and only ~97 ms of that
is the Winograd solver's GEMM. `ConvHipWinogradF2x3`
(`rocm6.4.4/patches/miopen/winograd-f2x3-hip-gfx803.patch`) transforms the
input into 4x4 tiles (4x the data of the input) in one kernel, writes them to
DRAM, runs a strided-batched GEMM over them, writes the result, and
transforms it back in a third kernel. Those round trips through DRAM are most
of the remaining ~85 ms.

**Approach.** One kernel per output tile block that transforms input tiles in
LDS, runs the 16 small GEMMs from LDS and registers, and transforms the
output before writing it. This is roughly what MIOpen's binary Winograd
kernels (`ConvBinWinogradRxS`) do as persistent kernels, and those are
disabled on gfx803 because they corrupt memory and hang the GPU (FINDINGS
section 2), so build it as ordinary HIP with ordinary launches. Forward and
backward-data both matter (training).

**Expected gain.** 30-60 ms per SD step (5-10%) if most of the transform
traffic goes away; training gains on both directions. Measure the split
first: time the SD 1.5 conv shapes with the solver forced
(`MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvHipWinogradF2x3`, empty
`MIOPEN_USER_DB_PATH`) against the time of its GEMM alone, which the shim's
call log gives in shape. `MIOPEN_DEBUG_CONV_HIP_WINOGRAD_F2X3=0` disables the
solver for comparison.

**Cost.** A MIOpen change: rebuilding MIOpen also rebuilds MIGraphX and
ONNX Runtime in the base image (~50 min), and users must clear their MIOpen
find-db once, or old records keep choosing other solvers.

**Verify.** All 17 SD 1.5 conv shapes forward and backward-data against
CPU float64, the 3-image VAE latent check, the UNet check, a training run.

### LLM-5: the unexplained part of a decoded token

**Why.** About 7 of the 37.6 ms per token of Qwen3 8B at a 5.2k-token context
are not explained by weight and KV reads (see "Where the time goes"). HIP
graphs are already in use: `GGML_HIP_GRAPHS` defaults to ON in ggml-hip at the
pinned version, the only architecture check that disables graphs applies to
NVIDIA GPUs older than Volta, and generation with
`GGML_CUDA_DISABLE_GRAPHS=1` is the same as without it (`llama-bench` tg128:
30.6 against 30.5 tokens/s). So the gap is GPU time, or synchronization, not
per-launch CPU cost.

**Approach.** Find where the time goes before choosing a fix: time the
per-token ops in isolation with `test-backend-ops perf` (`RMS_NORM`, `ROPE`,
`SET_ROWS` / `CPY` for the KV cache, `GLU`, `ADD`, `SOFT_MAX`, and
`FLASH_ATTN_EXT` at the decode shapes and context lengths), multiply by their
count per token, and compare with the gap. Small ops whose time is dominated
by launch and tail effects, rather than bytes moved, are candidates for fusion
into their neighbours.

**Expected gain.** Unknown; at most the ~7 ms (~19% generation) if all of it
turned out to be avoidable.

### SD-2: Winograd F(4x4, 3x3)

F(4x4, 3x3) needs 2.25 multiplies per output against F(2x2, 3x3)'s 4, so the
GEMM part (~97 ms per SD step) would shrink by ~44%, but the 6x6 tiles carry
2.25x the transform data of F(2x2)'s, and its transforms lose more precision
in fp32. Worth trying only once SD-7 has removed the transform round trips;
check precision against the CPU on every SD 1.5 shape.

### SD-1: Winograd weight gradients

`ConvHipWinogradF2x3` covers forward and backward-data. LoRA training freezes
the convolution weights and never needs backward-weights; full fine-tuning
or LoCon would, and falls back to MIOpen's GEMM path (im2col, then GEMM). The
weight-gradient form is an input transform, an output-gradient transform, a
batched GEMM accumulated over tiles, then the filter transform.

### SD-8: fused attention backward

Training's attention stores the score matrix and runs plain bmms in both
directions. A fused backward (recomputing scores per tile from the saved
log-sum-exp, as FlashAttention does) would cut attention's training time and
VRAM. Whether the VRAM saved lets 512 px training run without gradient
checkpointing, which costs one extra forward per step, is unknown. Large job;
measure attention's share of a training step first.

### LLM-3: decode kernels for 9-16 tokens

llama.cpp uses the decode kernels (mmvq) up to 8 tokens and the prompt
kernels (mmq) above. Whether mmvq is faster on gfx803 for 9-16 tokens, which
speculative decoding and parallel requests produce, is unmeasured.

### LLM-1b: SDWA losses

Patch 104's SDWA dot product makes 2-token decode 1-11% slower than the
compiler's code (q2_K, q6_K, q8_0), and lost 12-21% in the prompt kernels and
up to 2.6x for table-lookup operands before those were excluded. Why is not
known; the disassembly of the affected kernels against the generic build is
the place to start. Minor.

## Ruled out

Measured, and not worth pursuing unless something changes:

- **More norm and activation fusion.** GroupNorm, LayerNorm and SiLU together
  are ~2% of an SD step; the node's GroupNorm (1.0-1.35x) and GEGLU (1.6-1.7x)
  kernels already cover the largest pieces.
- **fp16 compute for SD.** fp16 runs at fp32's rate on this card, and its 3x3
  convolutions only became as fast as fp32 with the fp16 Winograd solver. fp16
  is worth its memory savings (SDXL training), not for speed.
- **Further integer dot-product work for decode.** K-quant decode matmuls are
  at ~80% of reachable bandwidth after patches 103 and 104.
- **DPP cross-lane reductions** (gfx8 has them). Reductions are a small share
  of every workload measured here.
- **HIP graphs for decode.** Already on by default and in use on gfx803;
  turning them off changes nothing measurable (LLM-5).
- **Routing fp16 GEMMs in every layout to the shim** (BLAS-4 below).
- **Redefining `C10_WARP_SIZE` for gfx8** (PT-1 below): a global macro change
  breaks kernels written for the old value.

## Done

### LLM-4: prompts through dequantize + fp32 SGEMM

[`106-gfx803-cublas-prompt.patch`](../rocm6.4.4/patches/ollama/llama-cpp/106-gfx803-cublas-prompt.patch)
sends quantized matmuls with 128 or more columns (prompt batches) to cuBLAS in
fp32 on gfx803, where the base image's shim runs them, instead of mmq's
emulated int8. Three things had to change. llama.cpp chose fp16 compute with
fp16 accumulation for quantized weights on every AMD GPU, which bypasses the
shim. Weight rows of 4096 or 14336 floats are multiples of 8 KiB, so the
weights are dequantized straight into rows padded by 16 floats (14336 x 512 x
4096: 2.8 -> 4.1 TFLOPS). And mmq was chosen before cuBLAS was considered.
Weights whose fp32 copy would exceed 512 MiB (output heads, which reach this
path only when every token's logits are computed) keep mmq: a 2.5 GB copy
next to the model ended in a GPU memory fault.

- Crossover, Qwen3 8B prompt tok/s by batch, cuBLAS vs mmq: 64: 132 vs 146,
  96: 130 vs 149, 128: 178 vs 154, 512: 222 vs 162.
- llama-bench pp512: Qwen3 8B 163.5 -> 222.8, Qwen2.5-Coder 7B 166.9 ->
  230.5; 8192 deep 127.6 -> 158.8 and 57.2 -> 64.1; Qwen3 8B pp12288 135.0 ->
  172.7.
- Ollama, ~5.2k-token prompt: Qwen3 8B 149 -> 215 tok/s, Qwen2.5-Coder 7B
  107 -> 136; generation unchanged.
- Precision: KL divergence against the CPU backend, Qwen2.5-Coder 7B 0.00152
  -> 0.00094, Qwen3 8B 0.0103 -> 0.0086. test-backend-ops MUL_MAT 1301/1301,
  MUL_MAT_ID 931/931, FLASH_ATTN_EXT 3982/3982.
- MoE models gain only through their dense layers (+4-5% on Qwen3.6
  35B-A3B); their experts keep mmq (LLM-6).

### LLM-7: DeltaNet conv-input concat

[`107-concat-transposed-src1.patch`](../rocm6.4.4/patches/ollama/llama-cpp/107-concat-transposed-src1.patch)
gives CONCAT along dim 0 with a transposed second operand (Gated DeltaNet's
convolution input, in Qwen3.5/3.6 and Qwen3-Next) a tiled kernel. The generic
kernel read the transposed operand with lanes 32 KiB apart: 27 ms per call,
0.83 s of a 2048-token batch of Qwen3.6 35B-A3B. Now 1.14 ms; Qwen3.6 pp2048
244 -> 270 tok/s at a 2048-token batch. Not specific to gfx803.

### BLAS-6: GEMMs with 1-8 columns

A PyTorch model that generates one token at a time multiplies each weight
matrix by 1 or 2 columns (2 with classifier-free guidance). rocBLAS ran the
fp16 calls (op(A) = T) on a Tensile kernel with a 128x32 macro tile, and the
shim ran the fp32 calls on its own tiles; with 2 columns almost all of each
tile was idle. `gfx803_sgemm_small_n_kernel` handles n <= 8 with op(A) = T
like a GEMV: each wavefront owns two rows of C, its lanes read K in 16-byte
loads, all columns' sums stay in registers, and a butterfly reduction ends
each row. When K is long enough, up to four wavefronts share a row and add
their sums through LDS, because a 1024-row weight otherwise leaves 3-4
wavefronts per SIMD. The shim now takes fp16 calls of this shape from
rocBLAS.

`F.linear` at batch 2 on the RX 590, before and after:

| Weight | fp16 | fp32 |
| --- | --- | --- |
| 1024 -> 1024 | 249 -> 14 us | 120 -> 24 us |
| 1024 -> 4096 | 263 -> 64 us | 261 -> 77 us |
| 4096 -> 1024 | 987 -> 44 us | 412 -> 82 us |
| 1024 -> 8194 | 355 -> 92-221 us | 507 -> 148 us |

In Chatterbox's 0.5B token model (30 Llama layers at batch 2), the linear
layers per token fall from 75.6 to 7.0 ms in fp16 and from 43 to 10.1 ms in
fp32. A whole token falls from 88 to 25.7 ms, and a 4.4 s sentence from 11.8
to 4.9 s; the rest is now CPU time, the Python decode loop.

The kernel sums in a different order than the tiles, so its results differ
from theirs in the last bits. Errors against a float64 CPU reference equal
the old path's in every layout tested (`linear`, misaligned rows, `addmm`
with beta, transposed activations, `bmm`; n from 1 to 10, fp16 and fp32).
Of the shim's 46 reference GEMMs, 44 stay bit-identical; the two with one
column that now take this kernel differ by 3e-7 relative. Chatterbox's
token model matches the CPU as before: fp32 logits 8.7e-7, fp16 5.6e-4.
`GFX803_SGEMM_SMALL_N=0` switches the kernel off.

### BLAS-5: TN, NT and TT layouts

`gfx803_sgemm_tile_g` in
[`gfx803_sgemm.h`](../rocm6.4.4/patches/sgemm-shim-batched/gfx803_sgemm.h)
serves the transposed layouts that `torch.nn.Linear` (TN) and its weight
gradient (NT) produce. NN keeps `gfx803_sgemm_tile_p_fast`. Three changes,
each measured on its own:

- **Transposed B staged like A in NN.** Four N-contiguous values per float4
  load and one LDS write, instead of lanes strided by ldb. SD 1.5's weight
  gradients go from 2.5-3.2 to 3.8-5.3 TFLOPS.
- **A 32-deep K slice for a transposed A.** Each row of A is then read in
  128-byte pieces instead of 64. The two other ways of staging a transposed A
  measured slower: single floats with 16 lanes along K, and the old float4
  mapping with a 16-deep slice.
- **Grouped tile order when A is larger than the 2 MiB L2.** Walking row tiles
  re-reads all of A from DRAM for every column tile. Groups of 8 column tiles
  that walk the row tiles together read it once per group. Applied to NN, the
  same grouping made large GEMMs slower, so NN is not grouped.

Which B staging and tile order win depends on how B's rows are aligned. With
rows at multiples of 256 bytes (nn.Linear's widths), single floats for K >=
2048 and grouped order measured best. Rows padded off that alignment
(llama.cpp's prompt GEMMs after LLM-4) measured best with float4 B in
row-tile order (8192 x 2048 x 2048: 2.1 -> 3.9 TFLOPS). TN GEMMs with A
within L2, and K <= 64, stay on `gfx803_sgemm_tile_p_fast`, which measured
the same or faster there.

**Leading dimensions that are multiples of 8 KiB** (2048 or 4096 floats) put
every row of a K-contiguous operand in one DRAM channel. 2048^3 TN runs at
3.4 TFLOPS against 4.95 for NN, and llama.cpp's 14336 x 512 x 4096 at 2.85.
Padding the leading dimension by 16 floats takes the same GEMMs to 3.6 and
4.1. Grouping made these shapes worse (TN 2048^3 2.9, NT 1.4), so they keep
the older kernels. The fix for them is a padded layout where the matrix is
allocated, which is part of LLM-4.

Results:

- **Correctness.** Bit-identical to the previous shim on 3000 random GEMMs
  (every layout, odd sizes and leading dimensions, offsets, batched,
  alpha/beta), the 46-case reference set and the 300-case PyTorch sweep
  (worst error against CPU float64 2.2e-6). A debug build ran the old and new
  kernels side by side on all 12,000+ GEMM calls of two SD 1.5 runs: 0 differ.
- **Kernel speed, TFLOPS** (old -> new, best of both measurement orders):

  | shape (M x N x K, layout) | old | new |
  | --- | --- | --- |
  | 2560 x 8192 x 320 TN (GEGLU proj) | 3.19 | 4.45 |
  | 5120 x 2048 x 640 TN | 2.80 | 4.40 |
  | 640 x 2048 x 2560 TN (feed-forward out) | 2.23 | 3.58 |
  | 1280 x 512 x 5120 TN | 1.97 | 3.20 |
  | 10240 x 512 x 1280 TN | 2.72 | 3.56 |
  | 320 x 2560 x 8192 NT (weight gradient) | 2.73 | 5.04 |
  | 640 x 5120 x 2048 NT | 2.94 | 5.29 |
  | 320 x 320 x 8192 NT (32x32 tile) | 1.58 | 2.50 |
  | 40 x 4096 x 4096 x3 NT (attention backward) | 1.69 | 2.55 |
  | 1000 x 3000 x 700 TT | 2.90 | 4.38 |
  | 4096 x 512 x 14336 TN, padded ld | 3.20 | 3.76 |

- **SD 1.5 inference** (ComfyUI, 20 steps, fp32, 512 px): 557 -> 526 ms per
  step with the node, 675 -> 644 without it. Without the node the latent is
  bit-identical to the old shim's. With the node it differs by at most 1.1e-4
  (relative L2 2e-6); see "Measuring" for why that is not a GEMM difference.
- **Training** (Sherlock, 40 steps, seed 42, shared find-db): 3.63 -> 3.37 s
  per steady step, avr_loss 0.204 -> 0.203, no GPU faults.

### BLAS-1, BLAS-2: pipelined 64x64 and 32x32 tiles

[`gfx803_sgemm.h`](../rocm6.4.4/patches/sgemm-shim-batched/gfx803_sgemm.h)
adds `gfx803_sgemm_tile_p_fast` (64x64) and `gfx803_sgemm_tile_p2_fast`
(32x32) for fp32 calls with B not transposed. Each thread issues the LDS reads
for the next K step before the multiply-adds of the current one, and with A
not transposed it fetches A with one float4 (64x64) or float2 (32x32) load.
Transposed-B calls and fp16 keep the previous kernels, which gained the same
LDS prefetch.

- Bit-identical to the previous shim: 46 of 46 reference cases and 300 of 300
  random GEMMs; worst error against a CPU float64 product 2.2e-6.
- torch fp32 matmul, ms (previous shim -> this one):

  | shape (M x N x K) | before | after |
  | --- | --- | --- |
  | 2048 x 2048 x 2048 | 5.22 (3.3 TFLOPS) | 3.56 (4.8) |
  | 4096 x 4096 x 4096 | 45.7 (3.0) | 34.1 (4.0) |
  | 4096 x 1280 x 320 | 1.19 | 0.86 |
  | 4096 x 320 x 1280 | 1.18 | 0.78 |
  | 512 x 1280 x 11520 (conv at 16x16) | 5.98 | 4.16 |
  | 128 x 1280 x 11520 (conv at 8x8, 32x32 tile) | 2.37 | 1.73 |
  | 4096 x 8 x 320 (LoRA down) | 0.064 | 0.052 |
  | 4096 x 320 x 8 (LoRA up) | 0.038 | 0.041 |
  | 4096 x 320 x 320, A transposed | 0.310 | 0.322 |
  | 2048 x 2048 x 2048, B transposed | 6.45 | 5.46 |

  These are plain `A @ B` on contiguous tensors, which reach the shim as NN;
  `torch.nn.Linear` reaches it as TN (BLAS-5).
- SD 1.5 LoRA training: 4.0 s per step against about 4.1 before.

### BLAS-4: fp16 GEMMs in every layout (evaluated, not adopted)

rocBLAS's fp16 kernels are slow on gfx803 only for op(A) = N, op(B) = T, which
the shim already takes. Routing the other three layouts to the shim's fp16
path made an SD 1.5 fp16 UNet step slower (633.5 -> 675.4 ms; rocBLAS ran
those layouts at 3.1-3.2 TFLOPS against the shim's 2.5-2.6 at the time), so
the shim leaves them to rocBLAS. The fast tiles are fp32-only; if they get
fp16 variants, this is worth re-measuring.

### SD-3, SD-4, SD-5: ComfyUI node

[`tools/ComfyUI-gfx803-attention/`](../tools/ComfyUI-gfx803-attention/)
replaces three operations in ComfyUI's SD 1.5 path with HIP kernels from
[`tools/`](../tools/):

- **Attention** (`gfx803_fused_attention.hip`): one pass with an online
  softmax, so the N x M score matrix is never stored. Hooks
  `F.scaled_dot_product_attention` and ComfyUI's split and optimized
  attention for head sizes 40, 64 and 80 (fp16 inputs are computed in fp32);
  160 stays on ComfyUI's path, where the fused kernel was slower. Calls that
  need gradients go to PyTorch.
- **GroupNorm** (`gfx803_group_norm_silu.hip`): one block per group, float64
  accumulation, float4 loads and stores; fp32, up to 128 channels per group.
- **GEGLU** (`gfx803_geglu.hip`): the split, exact-erf GELU and product in
  one pass.

Correct against CPU float64 (attention within 3.5e-6 relative, GroupNorm and
GEGLU within 1e-5, over odd shapes, 2-D/3-D and non-contiguous inputs); a
full ComfyUI txt2img differs from a CPU run's by at most 0.3/255. An SD 1.5
fp32 step at 512 px: 697 -> 567 ms. Per call (batch 2, 8 heads), ms:

| | PyTorch SDPA | ComfyUI split | node |
| --- | --- | --- | --- |
| self-attention 64x64 (D 40) | 58.1 (2.3 GB extra) | 49.7 | 26.3 (10 MB) |
| cross-attention 64x64 | 2.20 | 1.09 | 0.69 |
| self-attention 32x32 (D 80) | 4.68 | 3.95 | 3.61 |
| self-attention 16x16 (D 160) | 0.49 | 0.34 | 0.79, not used |

GroupNorm 0.194 -> 0.166 ms at (2, 320, 64, 64); GEGLU 1.075 -> 0.638 ms at
8192 x 2560. The node is not part of the images:
`tools/ComfyUI-gfx803-attention/install.sh` compiles the kernels (with the
base image's `hipcc` if the host has none) and copies the node into
`rocm6.4.4/ComfyUI/custom_nodes/`.

### PT-1: layer_norm

PyTorch's HIP layer_norm was wrong on gfx803: its device code reduces over
32-lane warps (`C10_WARP_SIZE` is 32 for every AMD target except gfx9) while
the host launched it with 64-wide blocks (`at::cuda::warp_size()`).
[`gfx803-layer-norm-warp-size.patch`](../rocm6.4.4/patches/pytorch/gfx803-layer-norm-warp-size.patch)
sizes layer_norm's launches by the compiled warp size: native
layer_norm matches a CPU float64 reference in 80 of 80 runs, all three
gradients included.
Redefining `C10_WARP_SIZE` as 64 for gfx8 also fixed layer_norm but broke
group_norm's weight and bias gradients for more than 128 rows. Details:
FINDINGS section 5.

### LLM-1: SDWA dot product in the decode kernels

[`104-gfx803-sdwa-dp4a.patch`](../rocm6.4.4/patches/ollama/llama-cpp/104-gfx803-sdwa-dp4a.patch)
computes `ggml_cuda_dp4a` with SDWA byte-select multiplies in the decode
kernels (mmvq, flash attention). It frees registers: decode kernels above 128
VGPRs 56 -> 1, spills 2 -> 0, and 4-8 token decode matmuls up to 1.6x faster
(q2_K 8 tokens 2137 -> 1324 us, q3_K 4 tokens 809 -> 569, q8_0 4 tokens 426 ->
347). Single-token matmuls are unchanged and 2 tokens lose 1-11% (LLM-1b).
Applied everywhere, SDWA was slower in the prompt kernels (q8_0 tiles 21.6 ->
26.2 ms at 512 tokens, which the IQ types use too) and for table-lookup
operands (IQ4_XS decode up to 2.6x), so those call `ggml_cuda_dp4a_plain`, the
compiler's byte-wise product, and match the unpatched build. Ollama at a
5.2k-token context: Qwen3 8B and Qwen2.5-Coder 7B (Q4_K_M) generate 6-7%
faster (24.9 -> 26.6, 28.4 -> 30.5 tok/s), prompts and Qwen3.5 9B IQ4_XS
unchanged; output bit-identical. test-backend-ops: MUL_MAT 1301/1301,
MUL_MAT_ID 931/931, FLASH_ATTN_EXT 3982/3982.

### LLM-2: q5_K decode

[`105-gfx803-q5k-decode.patch`](../rocm6.4.4/patches/ollama/llama-cpp/105-gfx803-q5k-decode.patch)
reads q5_K's d/dmin and scales with one 16-byte load. With 104, the q5_K
matmul (4096x14336) goes from 233 to 224 us for one column and from 287 to
278 us for two.

## How to work on this

### Measuring

- **Profilers record nothing on the images' default runtime.** The images
  preload AMD's stock HSA runtime, and neither `rocprof` nor `torch.profiler`
  sees any kernels with it. The patched runtime (`GFX803_RUNTIME=patched`)
  works with profilers but page-faults in some workloads, and a fault storm
  can freeze the desktop: profile on it only with the desktop stopped.
- **Per-module timing without a profiler.** Register forward pre- and
  post-hooks on every leaf module (and on attention modules) that record a
  `torch.cuda.Event`; run one warm-up forward, then one timed forward, and sum
  `start.elapsed_time(end)` per module class. This is how the SD breakdown
  above was measured. Events do not synchronize, so the overhead is small.
- **Which GEMMs a workload issues.** `GFX803_SGEMM_SHIM_VERBOSE=N` makes the
  shim print its first N calls (`sgemm`, `gemm_ex`, strided-batched forms)
  with layout, m, n, k, batch and its decision. Sum 2mnk x batch per layout
  for the FLOP shares.
- **Warm up first.** The first pass through MIOpen with a fresh find-db runs
  MIOpen's Find, which times every candidate solver: one SD UNet pass then
  logs about 4x the GEMM FLOPs of a real step. In training, each new bucket
  size triggers Find again (~50 s on that step), so judge training speed by
  steady steps, not tqdm's average.
- **SD harness.** ComfyUI can run in-process: put `/opt/ComfyUI` on
  `sys.path`, set `sys.argv` to ComfyUI's flags, call
  `comfy.options.enable_args_parsing()`, load with
  `nodes.CheckpointLoaderSimple().load_checkpoint(...)`, and use
  `model.model.diffusion_model` directly or `nodes.KSampler().sample(...)`.
  Use `--use-split-cross-attention` for anything with gradients (ComfyUI's
  default sub-quadratic attention writes `exp(..., out=...)` in place and
  cannot be differentiated), and `--force-fp32` to match the service. A 512 px
  forward and backward without gradient checkpointing does not fit in 8 GB.
- **GEMM speed.** Time `A @ B` with CUDA events after a few warm-up calls;
  build transposed operands as `.t()` views to reach each rocBLAS layout.
  Two effects of 5-15% hide in back-to-back A/B timing. The variant timed
  first in each round runs faster on some shapes, whatever its code. Output
  buffers at different physical addresses also run at different speeds
  (DRAM channel mapping). Time both variants into the same buffer, in both
  orders, and keep the best of each.
- **Which GEMM call differs.** A debug shim that includes the old and new
  headers in two namespaces runs both kernels on every intercepted fp32
  call, compares them byte for byte, and prints the parameters of any call
  that differs. It answers whether an end-to-end difference comes from the
  shim at all.
- **ComfyUI end-to-end comparisons.** Compare runs in separate processes. In
  one process, a second identical sampling differs from the first (latent
  max 2.4e-3 without the node, 6.4e-3 with it), with either shim. Separate
  processes with the same binaries are bit-identical. With the node
  installed, the result also depends on GPU timing. The latent differs by
  1e-4 when only the GEMMs get faster, even though every GEMM call is
  bit-identical (debug shim) and each node kernel is deterministic on its
  own (same inputs, eight different dirty-memory states). The source of this
  dependence is not found yet.
- **llama.cpp kernels.** `test-backend-ops perf -o MUL_MAT -b ROCm0 -p
  'type_a=q4_K,.*n=512,'` (the `-p` filter is a regex over the case string).
  To compare two builds fairly, run the same `test-backend-ops` binary against
  each build's `libggml-hip.so`.
- **Ollama end to end.** A fixed ~5k-token prompt, `temperature 0`,
  `top_k 1`, `seed 0`, 128 tokens with log-probabilities, through
  `/api/generate`; compare tokens and log-probabilities between images, and
  read `prompt_eval_duration` and `eval_duration` for the speeds.
- **Register use.** Extract the gfx803 code object (`.hip_fatbin` section,
  `clang-offload-bundler --unbundle`, one bundle per translation unit) and read
  `.vgpr_count` and `.vgpr_spill_count` with `llvm-readelf --notes`.

### Building and iterating

- **Shim.** Compile in the base image without rebuilding it:
  `hipcc -O2 -fPIC -shared --offload-arch=gfx803 -I/opt/rocm/include
  sgemm_shim.cpp -o libtest.so -L/opt/rocm/lib -lrocblas -ldl`, then mount
  the result over `/opt/rocm/lib/libgfx803_sgemm_shim_batched.so` in a test
  container. `GFX803_SGEMM_MIN_TILES64` / `GFX803_SGEMM_MIN_TILES32` override
  tile choice (0 = always, -1 = never), `GFX803_SGEMM_SHIM_DISABLE=1` passes
  everything to rocBLAS. In the image the shim builds in its own stage, so a
  shim change rebuilds only that stage.
- **llama.cpp.** Patches in `rocm6.4.4/patches/ollama/llama-cpp/` apply with
  `git apply` in `Dockerfile-Ollama`, each checked by a marker grep. To build
  `test-backend-ops` from the patched tree, the Ollama builder's llama.cpp
  needs Ollama's compat sources (`llama/compat/*.cpp`, `*.h`) added to its
  `src/CMakeLists.txt`, or it fails on missing `llama-ollama-compat.h` and
  then on undefined compat symbols. Run the test binaries in the base image
  (it has the gfx803 rocBLAS) with `LD_LIBRARY_PATH` pointing at the build's
  `bin/`. Editing a patch file's header text reruns the image's llama.cpp
  build (~12 min).
- **PyTorch.** Patches in `rocm6.4.4/patches/pytorch/`, each with a `.sh`
  driver that applies it with `git apply` and checks a marker. Any change in
  that folder, header text included, reruns the PyTorch build: ~35 min with a
  warm ccache, hours without.
- **MIOpen.** A change in `rocm6.4.4/patches/miopen/` rebuilds MIOpen, then
  MIGraphX and ONNX Runtime (~50 min). Old find-db records keep their solver
  choices until the user find-db is cleared.

### Verifying a change

Run what the change can affect, on the image's own binaries:

- **Shim:** bit-identity against the previous shim over a fixed set of GEMMs
  (all transposes, plain and batched, beta 0 and non-zero, odd sizes) and a
  few hundred random ones (odd leading dimensions, pointer offsets, batch
  strides, alpha/beta), plus their error against a CPU float64 product.
- **SD:** `rocm6.4.4/training/gfx803_unet_check.py` (ops forward and backward,
  then the full UNet, against the CPU); a ComfyUI txt2img compared with a CPU
  run of the same seed (latent relative L2 ~1e-5, image within 1/255); for
  convolution changes, every SD 1.5 conv shape forward and backward-data
  against CPU float64 and the VAE latent check (FINDINGS section 2).
- **Training:** 20 `gfx803-train` steps; avr_loss 0.15-0.2 is healthy, 3-4
  means broken GPU math.
- **llama.cpp:** `test-backend-ops test -o MUL_MAT`, `MUL_MAT_ID`,
  `FLASH_ATTN_EXT` (or the full suite; its only known failures are
  `SOLVE_TRI` cases larger than 64x64) against the CPU backend, and Ollama
  greedy output against the previous image on a Q4_K_M and an IQ4_XS model.
- **Stability:** no `VM fault` / `ring timeout` lines in `journalctl -k` over
  the run; for kernels that run for long periods, the 45-minute block replay
  under full load (FINDINGS section 3).

### Lessons that cost time

- **Measure every path a change touches.** Patch 104 was a win in the decode
  kernels and a loss in the prompt kernels and for one operand type; the
  first version applied it everywhere and slowed the IQ4_XS model's prompts by
  8%.
- **Do not change a global macro to fix one kernel.** Redefining
  `C10_WARP_SIZE` fixed layer_norm and broke group_norm's backward, and was
  the likely cause of a GPU memory fault during training.
- **Speed claims from microbenchmarks did not survive end-to-end checks.**
  Several kernel-level gains reported for this work shrank or vanished in a
  real step (the shim's +50% on large GEMMs is ~2% of a training step).
  Quote end-to-end numbers next to kernel numbers.
- **Plain `A @ B` benchmarks miss the hot layout.** Linear layers are TN;
  benchmark with the layouts the workload's call log shows.
- **Power-of-two leading dimensions are their own case.** A 2048^3 or
  4096-wide benchmark measures DRAM channel conflicts, not the kernel. A tile
  order that won on SD's shapes lost 15-55% on those. Always include padded
  and real-model leading dimensions next to the square case.
