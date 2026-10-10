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
with `--use-split-cross-attention` and the gfx803 node: 566 ms (697 without
the node). Time per module class, from CUDA events around every UNet module's
forward:

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
batch 1, gradient accumulation 4) takes about 4.0 s per step at steady state.
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

Prompts run through mmq, which emulates int8 dot products: q4_K at 512 tokens,
4096x14336, takes 24.6 ms, about 2.45 effective TOPS, a third of the fp32 FMA
rate. Qwen3 8B reads a 5.2k-token prompt at ~149 tok/s and Qwen2.5-Coder 7B at
~107. A 12,288-token prompt, the size of a coding agent's first request, takes
92 s (133 tok/s; unmodified llama.cpp took 18 minutes, at 11.4 tok/s).

## Status

Open targets in recommended order, then the rest.

| Target | Area | Expected gain | Status |
| --- | --- | --- | --- |
| BLAS-5 | Shim: TN and NT layouts at NN speed | SD inference ~11% per step, training similar; enables LLM-4 | Open, first |
| LLM-4 | llama.cpp prompts through dequantize + fp32 SGEMM | ~1.5x prompt speed for dense K-quant models (after BLAS-5) | Open |
| SD-6 | Register-tiled fused attention (and llama.cpp's fp32 FA tile kernel) | SD inference ~12% per step; deep-context prompts | Open |
| SD-7 | Fused Winograd F(2x2, 3x3) | SD 30-60 ms per step (5-10%), training too | Open |
| LLM-5 | The unexplained ~7 ms per decoded token | unknown, up to ~19% generation | Open, measure first |
| SD-2 | Winograd F(4x4, 3x3) | unmeasured; only after SD-7 | Open |
| SD-1 | Winograd weight gradients | full fine-tuning and LoCon only | Open |
| SD-8 | Fused attention backward | training attention time and VRAM | Open, large |
| LLM-3 | mmvq vs mmq for 9-16 tokens | speculative/parallel decoding | Open |
| LLM-1b | SDWA dot product losses (2-token decode) | 1-11% on 2-token decode | Open, minor |
| BLAS-1, BLAS-2 | Shim: pipelined 64x64 and 32x32 tiles | large NN GEMMs 2.8-3.3 -> 3.9-4.8 TFLOPS | Done |
| BLAS-4 | Shim: fp16 GEMMs in every layout | slower than rocBLAS | Evaluated, not adopted |
| SD-3, SD-4, SD-5 | ComfyUI node: fused attention, GroupNorm, GEGLU | SD step 697 -> 567 ms | Done, optional install |
| PT-1 | PyTorch layer_norm without the group_norm route | correctness (and ~1%) | Done |
| LLM-1 | llama.cpp SDWA dot product in decode kernels (patch 104) | Q4_K_M generation +6-7% | Done |
| LLM-2 | llama.cpp q5_K decode (patch 105) | q5_K decode ~4% | Done |

## Open targets

### BLAS-5: shim TN and NT at NN speed

**Why.** Every `torch.nn.Linear` reaches the shim as `sgemm` TN: 40% of SD
inference GEMM FLOPs and 172 ms of a 566 ms step, at ~2.7 TFLOPS. Training
adds NT (26%) for the backward pass. The pipelined fast tiles (BLAS-1/2) only
help op(B) = N, and their TransA variant gained nothing (2048^3 with A
transposed: 2.92 TFLOPS before and after; NN reaches 4.8). NT calls use the
older `gfx803_sgemm_tile_p` (3.1-3.4 TFLOPS). llama.cpp's fp32 path (LLM-4)
is TN as well.

**Where.** `rocm6.4.4/patches/sgemm-shim-batched/gfx803_sgemm.h`:
`gfx803_sgemm_tile_p_fast<P, TransA, E>` (its `TransA` branch of `fetch`) and
the dispatch in `gfx803_gemm_strided_batched`, which sends `transB` calls to
the old kernel.

**Approach.** With A transposed, A's contiguous dimension is K. The current
TransA fetch gives each thread one float4 along K for one row (4 lanes per
row, rows strided by lda) and then stores it into the M-major LDS tile with
four scalar writes, which also 2-way bank-conflict (row stride 68 floats).
Options: stage A in LDS K-major (as it arrives) and read it in the
multiply-add loop with a layout that keeps the ping-pong prefetch; or
transpose through registers so the LDS writes stay vectorized. NT is the
mirror image for B: today's fast B staging walks K, which is B's strided
dimension when B is transposed, so it needs the same treatment for B. Keep
the accumulation order identical, so the result stays bit-identical to the
reference kernel.

**Expected gain.** TN at ~4.5 TFLOPS: SD inference 172 -> ~105 ms of linear
time, about -65 ms per step (~11%). Training gains on its TN and NT half. It
is also the prerequisite for LLM-4.

**Verify.** Bit-identity against the current shim over the reference set and
a random sweep (every layout, odd sizes and leading dimensions, offsets,
batched, alpha/beta: see "Verifying a change"); then the SD, training and
Ollama checks.

### LLM-4: llama.cpp prompts through dequantize + fp32 SGEMM

**Why.** Prompt processing is the slowest part of using local coding agents
here, and it runs on emulated int8 (mmq, ~2.45 effective TOPS). llama.cpp
also has a path that dequantizes the weights and calls cuBLAS
(`ggml_cuda_op_mul_mat_cublas` in `ggml-cuda.cu`). Its fp32 branch calls
`cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, ...)`, which hipBLAS sends to
`rocblas_sgemm` and the image's preloaded shim intercepts (the Ollama image is
built on the base image and preloads it). This route was measured on
2026-10-03 at 1.84 TFLOPS and rejected; the shim is faster now, and after
BLAS-5 it should run at ~4.5.

**Estimate.** q4_K, 4096x14336, 512 tokens: 60 GFLOP. At 4.5 TFLOPS that is
~13.4 ms, plus ~2 ms to dequantize the weight to fp32 (235 MB written and
read), against mmq's 24.6 ms: about 1.5-1.6x. At today's TN speed (2.9) it
would be ~22.7 ms, barely ahead; this is why BLAS-5 comes first. fp32 GEMMs
are also more precise than mmq, which quantizes activations to 8 bits.

**Approach and pitfalls.**
- `ggml_cuda_should_use_mmq` (`mmq.cu`) chooses mmq; the build option
  `GGML_CUDA_FORCE_CUBLAS` turns it off globally, which is enough for a first
  measurement. A gfx803 rule would route by batch size (mmvq stays for
  decode).
- Check which compute type the cuBLAS path picks for quantized weights on
  gfx803. An earlier trial of this route used fp16 with fp16 accumulation
  (`CUBLAS_COMPUTE_16F`, +10%, rejected for precision); the target is the fp32
  branch.
- The dequantized fp32 copy of the largest weight needs ~235 MB of temporary
  VRAM (Qwen3 8B's 4096x14336 FFN), which must fit beside the model and KV
  cache in 8 GB.
- MoE models go through `MUL_MAT_ID`, a different path; dense models first.

**Verify.** test-backend-ops `MUL_MAT` and `MUL_MAT_ID` against the CPU;
greedy Ollama output against the previous image (tokens may legitimately
change at near-ties, because fp32 differs from mmq's int8 rounding, so
compare KL divergence and top-token agreement against the CPU backend
instead of exact identity); prompt tok/s at 0 and 8k depth.

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
  `torch.nn.Linear` reaches it as TN (see BLAS-5).
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
the host launched it with 64-wide blocks (`at::cuda::warp_size()`), so the
image routed layer_norm through group_norm.
[`gfx803-layer-norm-warp-size.patch`](../rocm6.4.4/patches/pytorch/gfx803-layer-norm-warp-size.patch)
sizes layer_norm's launches by the compiled warp size, and the route is gone:
native layer_norm matches a CPU float64 reference in 80 of 80 runs, all three
gradients included, and an SD step is ~1% faster than through the route.
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
