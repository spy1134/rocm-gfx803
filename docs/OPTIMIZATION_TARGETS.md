# Optimization targets for gfx803

Performance work on the `rocm6.4.4/` line: Stable Diffusion 1.5 and SDXL in
ComfyUI, SD 1.5 LoRA training with sd-scripts, llama.cpp through Ollama (dense
and MoE models) and the decode-shaped GEMMs of Chatterbox's TTS, on Polaris
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
| Power | 185 W board limit, which is also the most the driver accepts without overdrive | GEMM and attention load hold the clock at ~1.34 GHz (173-176 W): ~6.2 TFLOPS of multiply-adds sustained. The first ~0.1 s of load runs at the 1.56 GHz boost, so short timings read 10-20% fast. |
| fp32 multiply-add | `v_mac_f32` / `v_mad_f32` full rate; the fused `v_fma_f32` a quarter of it (894 against 3,540 G operations per second) | Write `a * b + c`, never `fmaf()`: the compiler contracts it to `v_mac_f32` because fp32 denormals are flushed on gfx8. The fused form remains in division and transcendental sequences, outside inner loops. |
| fp16 | same rate as fp32 | No packed fp16 math (`v_pk_*`): fp16 halves memory traffic and storage, not arithmetic time. |
| Integer dot products | none | No `v_dot4`/`dp4a`. Quantized dot products are ALU sequences; 24-bit integer multiplies (`v_mul_i32_i24`) are full rate, and SDWA byte/word operand selects are free. |
| Matrix cores | none | Every GEMM is plain FMA from registers. |
| Memory | 256 GB/s GDDR5 nominal, ~225 GB/s reached by streaming kernels (q8_0 and f16 single-token matmuls: 222-229 GB/s) | Single-token LLM decode, norms and elementwise ops are bound here. 28 FLOP per byte separates compute-bound from memory-bound work. |
| DRAM channels | 16 | Power-of-two leading dimensions can put a wavefront's accesses on one channel (partition camping; it cost the shim 20x). |
| Per CU | 64 KiB LDS; 4 SIMDs with 64 KiB of VGPRs each (256 VGPRs per lane); 64-lane wavefronts | Up to 128 VGPRs a SIMD holds 2+ waves, above it one; above 32 KiB of LDS per block a CU holds one block. |
| L2 | 2 MiB | |
| VRAM | 8 GB, shared with the desktop if the card drives it | Desktop framebuffers are invisible to ROCm's free-memory figure. |
| Host link | PCIe 3.0 x16, ~13 GB/s | Bounds anything that streams weights from system RAM (MoE experts, offloaded layers). |
| Kernel dispatch | back-to-back empty kernels take 4.3 us each, 2.65 us without the vector-L1 invalidate the rebuilt HIP runtime issues before every dispatch (FINDINGS section 4); the same in replayed HIP graphs | Workloads of many small kernels pay per launch: an MoE token issues ~1,450 dispatches, an fp16 SD 1.5 step ~1,430. Fusing kernels saves dispatches as well as bytes (RT-1). |

The rest of the family (RX 470/480/570/580: 32-36 CUs at lower clocks, same
256-bit bus) has the same balance, so the conclusions here should carry over,
scaled by clock. Polaris 11/12 (RX 460/550/560: 8-16 CUs, 64- or 128-bit
bus) is untested.

What follows from this: the card does dense fp32 multiply-add well, as long as
the data is reused from registers and LDS. Almost everything expensive in
these workloads is matrix multiplication: linear layers, convolutions (as
Winograd plus a batched GEMM), attention (two GEMMs around a softmax) and LLM
prompt processing. Making all of it run like the shim's best GEMM kernel does
is the main lever. The pipelined tiles now cover the NN, TN and NT layouts,
fp32 and fp16 (BLAS-1, BLAS-2, BLAS-4, BLAS-5); shapes whose leading
dimension is a multiple of 8 KiB still run on older kernels.

## Where the time goes

### SD 1.5 inference

One denoising step, fp16 UNet (ComfyUI's default on this card), 512 px,
classifier-free guidance (batch 2), with the gfx803 node and a MIOpen
find-db made by the current build: 342 ms (392 without the node). Time per
module class (`tools/unet_breakdown.py`: CUDA events around every UNet
module's forward, each module's time exclusive of its children):

| What | Time | How it runs | Efficiency |
| --- | --- | --- | --- |
| Convolutions | 127 ms (37%) | MIOpen `ConvHipWinogradF2x3`: input/output transforms in HIP kernels around a strided-batched GEMM in the shim (NN, 438 GFLOP per step) | transforms 26 ms by kernel time, the GEMM the rest |
| Linear layers (attention projections, feed-forward, proj in/out) | 108 ms (32%) | shim fast tiles with fp16 storage, TN layout (all 184 linears per step) | ~4.2 TFLOPS |
| Attention, excluding its projections | 79 ms (23%) | node's multi-wave fused kernel reading the fp16 projections in place, 32 calls per step; 64x64 self-attention 12.5 ms per call | ~3.2 TFLOPS (252 GFLOP) |
| GroupNorm, LayerNorm, SiLU | 10 ms (3%) | | |
| Residuals, reshapes, launch gaps | ~14 ms (4%) | estimate: step time minus the hooked modules | |

At fp16 only the node's attention runs: its GroupNorm and GEGLU kernels take
fp32 inputs, so the norms and GEGLU's elementwise passes stay on PyTorch
(SD-10).

By kernel (PyTorch profiler, 353 ms of kernel time, measured before the
two-wavefront TN tile and before SD-9 took the attention from 91 to 79 ms per
step): the shim's GEMMs 202 ms (57%: linears, the Winograd solver's batched
GEMMs and 1x1 convolutions, ~917 GFLOP at ~4.5 TFLOPS on average), fused
attention 87 ms,
elementwise kernels, norms and casts ~38 ms, the Winograd filter transform
17 ms and its input and output transforms 9 ms. The filter transform runs on
every call (48 per step) although the weights do not change between steps;
see "Ruled out".

A find-db written before a build added `ConvHipWinogradF2x3` for fp16 keeps
its fp16 3x3 records on `GemmFwdRest`, and convolutions then take about twice
as long.

End to end, a warm 20-step 512 px image through the comfyui image's service
(`--lowvram --use-split-cross-attention`, fp16 UNet) takes 8.5 s with the node
and 10.5 s without it. The 20 UNet steps are 6.8 and 7.8 s of that; the other
1.7 and 2.7 s (text encoder, VAE decode, `--lowvram` model moves, ComfyUI
itself) have no breakdown (SD-11).

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

### SDXL inference

Pony V6 (SDXL), 1024 px, CFG batch 2, fp16 UNet, the gfx803 node: one UNet
step 2.75 s. By kernel, before the two-wavefront TN tile (BLAS-7; step
2.98 s then): the shim's GEMMs 2.23 s (75%), fused attention 0.50 s (17%),
elementwise and norms 0.18 s, Winograd transforms 0.04 s; the fused
attention is now 0.42 s (SD-9). At sustained clocks the large linears run
at 5.0-5.3 TFLOPS (10240x2048x1280 10.2 ms), the 1280-wide ones with few
tiles stay on the 64x64 tile at 4.1-4.6 (see BLAS-7), and the
cross-attention K/V projections of the 77-token context (n = 154, 1.1-2.5
TFLOPS) lose about 30 ms per step.

The VAE decode at 1024 px is the other cost. With `--bf16-vae` its
convolutions run on im2col GEMMs (there is no bf16 Winograd), and ComfyUI
falls back to tiled decoding when the regular decode runs out of memory,
which it does on some images and not others: 20-step images took 83 and
99 s end to end with the same 67 s of sampling, at 3.36 s per UNet step (now
2.75 s, about 55 s of sampling; the end-to-end times are not re-measured).
With `--fp32-vae` the decode uses the F(2x2,3x3) solver and the shim and took
86 s with tiling (decode alone, tiled: bf16 24.6 s, fp32 12.2 s; outputs
within 0.03 of each other on a 0-1 scale). MIOpen's first Find over the bf16
VAE shapes costs several minutes once, because its naive direct solver takes
up to 1.5 s per call on the large decoder shapes. The decode is SD-12.

### SD 1.5 LoRA training

A run with `gfx803-train`'s defaults (fp16 mixed precision, 512 px buckets,
gradient checkpointing, batch 1, gradient accumulation 4) takes about 2.5 s
per step at steady state, 1.7 s without gradient checkpointing; in fp32,
2.9 and 2.1 s (TRAIN-1). Attention runs on the fused forward and backward
kernels (SD-8). No breakdown of the step by phase is recorded (TRAIN-3). Over
a full-length run fp16 and fp32 give the same loss and images (TRAIN-2).
GEMM FLOPs by layout for one UNet forward and backward with frozen weights,
which is what LoRA training's base model does, at 384 px (fp32; the shares
do not depend on the precision):

| Layout | Share |
| --- | --- |
| NN | 50% |
| TN | 24% (forward linears, attention) |
| NT | 26% (backward of convolutions, linears and attention) |

### llama.cpp decode (token generation)

Qwen3 8B Q4_K_M with a 5.2k-token context generates 27.81 +- 0.09 tok/s, 35.96 ms
per token (31.16 +- 0.09 tok/s, 32.09 ms at zero context depth), measured
before the stale-L1 fix (2.3-2.7% of a 7B model's decode) and patch 111
(+1-2%). The K-quant decode kernels read the 4.87 GB of weights at ~198 GB/s
(24.54 ms), against the 222-229 GB/s that q8_0 and f16 single-token matmuls
reach (21.6 ms at 225 GB/s); the q8_0 KV cache takes 4.42 ms, 1.8 ms at that
rate. LLM-5 splits the ~12.5 ms above that bound: flash-attention decode at
~90 GB/s, 2.38 ms of unquantized F16 `attn_v` projections at ~127 GB/s, the
K-quant kernels' ~2.9 ms, 2.97 ms across 581 small-kernel launches and ~1.9 ms
idle between kernels. MoE generation with experts in system RAM: LLM-8.

### llama.cpp prompt processing (prefill)

Batches of 128 tokens or more run as fp32 GEMMs on the shim (LLM-4); smaller
ones, and MoE expert matmuls, run through mmq, which emulates int8 dot
products at about 2.4 effective TOPS. Qwen3 8B reads a 5.2k-token prompt in
Ollama at ~215 tok/s and Qwen2.5-Coder 7B at ~136. A 12,288-token prompt, the
size of a large coding context request, takes 71 s (173 tok/s; unmodified
llama.cpp took 18 minutes, at 11.4 tok/s). MoE prompts: LLM-6; images:
LLM-10.

## Status

Open targets in recommended order, then the done ones.

| Target | Area | Expected gain | Status |
| --- | --- | --- | --- |
| LLM-6 | MoE prompts: expert copies, MUL_MAT_ID, dense parts | copies overlap compute since patch 108 (pp2048 295 -> 367 tok/s, against 376 with no copies at all); the rest is compute, with no breakdown | Open, profile first |
| RT-1 | Cost of the vector-L1 invalidate before every dispatch | 1.65 us per dispatch: ~2.4 ms of an MoE token (~6%), 2.3-2.7% of dense decode, ~2.4 ms of an SD 1.5 step | Open, unmeasured |
| LLM-5 | Dense decode above the bandwidth bound | per 36 ms token: K-quant kernels ~2.9 ms, flash-attention decode ~2.6 ms at 5k depth, F16 `attn_v` ~1 ms, launches ~1 ms by fusion; the same kernels make up MoE generation's GPU part | Open, measured before the stale-L1 fix and patch 111 |
| LLM-8 | MoE generation with experts in system RAM | shared expert overlaps the CPU experts since patch 112 (Ollama 25.3 -> 26.2 tok/s); left: the GPU split's kernels and dispatches, the fp32 router ~0.5 ms, the hand-offs, the CPU experts (at most ~3.5 ms) | Open, measured before patches 111 and 112 |
| SD-10 | fp16 GroupNorm/SiLU and GEGLU in the node | the node's kernels take fp32 only, so ComfyUI's fp16 default runs neither; a few ms of a 342 ms step | Open, unmeasured |
| SD-11 | ComfyUI time outside the UNet | 1.7 s of an 8.5 s warm SD 1.5 image (2.7 of 10.5 s without the node) | Open, unmeasured |
| SD-12 | SDXL VAE decode at 1024 px | 12-25 s of an 83-99 s image, by VAE precision | Open, measured end to end |
| SD-2 | Winograd F(4x4, 3x3) in the gfx803 HIP solver | estimate ~25 ms (7%) of an fp16 SD 1.5 step, more at SDXL; precision and small latents unknown | Open, estimate |
| TRAIN-3 | Where a training step goes | unmeasured; fp16 NT GEMMs still run on the older tiles | Open, profile first |
| BLAS-7 | Shim GEMM efficiency at SD 1.5 / SDXL shapes | fp16 dispatch fixes, fp16 widening, a one-wavefront tile for batched NN and a two-wavefront TN tile took the SD 1.5 step 415 -> 355 ms, SDXL 3.34 -> 2.84 s; left: a faster main tile, a 64x64 two-wavefront tile for few-tile TN calls | Open, measured |
| SD-9 | SDXL attention | P now moves between lanes inside the multiply-adds (DPP): SDXL step attention 0.50 -> 0.42 s (step 2.84 -> 2.75 s), SD 1.5 91 -> 79 ms (step 355 -> 342 ms); self-attention at ~4 TFLOPS against ~5.4 for its instruction mix | Open, measured |
| LLM-10 | Image prompts | a 1080p image (~2000 tokens) 56 s, against ~15 s for 2000 text tokens (Qwen3.5 9B) | Open, unmeasured |
| SD-7 | Fused Winograd F(2x2, 3x3) | fp16 SD at most ~9 ms per step (2%); fp32 and training unmeasured | Open, low |
| SD-1 | Winograd weight gradients | full fine-tuning and LoCon only | Open |
| LLM-3 | mmvq vs mmq for 9-16 tokens | speculative/parallel decoding | Open |
| LLM-1b | SDWA dot product losses (2-token decode) | 1-11% on 2-token decode | Open, minor |
| TRAIN-1 | fp16 LoRA training against fp32 | 2.87 -> 2.47 s/step, 5.8 -> 3.6 GB; without checkpointing 2.07 -> 1.73 s/step, 7.1 -> 4.9 GB | Done (fp16 is `gfx803-train`'s default) |
| TRAIN-2 | fp16 LoRA training over a full-length run | checks `gfx803-train`'s default: same loss and images as fp32 over 1200 steps | Done |
| SD-8 | Fused attention backward for training | SD1.5 LoRA 512 px, fp32: 3.37 -> 2.87 s/step and 7.6 -> 5.6 GB; 2.03 s/step without gradient checkpointing | Done |
| SD-6 | Multi-wave fused attention | SD 1.5 fp16 step attention 129 -> 91 ms (step 395 -> 360 ms); SDXL self-attention 2.9x | Done |
| PT-2 | PyTorch kernels that mix the host's 64-lane warp size with the device's 32 | correctness: wrong results or a GPU reset | Done: four kernels fixed, the audit's other suspects match the CPU; PyTorch's OpInfo sweep left |
| LLM-9 | Decode converts a shared input to q8_1 once (patch 111) | generation +1-2% (Qwen3 8B 30.1 -> 30.3-30.8 tok/s) | Done |
| LLM-4 | llama.cpp prompts through dequantize + fp32 SGEMM (patch 106) | Qwen3 8B prompt 149 -> 215 tok/s in Ollama | Done |
| LLM-7 | DeltaNet conv-input concat (patch 107) | Qwen3.6 35B-A3B pp2048 244 -> 270 tok/s | Done |
| BLAS-4 | Shim: fp16 operands on the fast tiles, all layouts | fp16 SD 1.5 step 602 -> 483 ms; 31 GEMM cases median 1.5x | Done |
| BLAS-6 | Shim: GEMMs with 1-8 columns (decode-shaped), fp16 and fp32 | Chatterbox TTS token model 88 -> 25.7 ms per token | Done |
| BLAS-5 | Shim: TN, NT and TT layouts | SD step 557 -> 526 ms, training 3.63 -> 3.37 s/step | Done |
| BLAS-1, BLAS-2 | Shim: pipelined 64x64 and 32x32 tiles | large NN GEMMs 2.8-3.3 -> 3.9-4.8 TFLOPS | Done |
| SD-3, SD-4, SD-5 | ComfyUI node: fused attention, GroupNorm, GEGLU | SD step 644 -> 526 ms | Done, optional install |
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

**Where the time goes.** A batch copies every layer's experts (about 20 GB)
over PCIe, once per batch, so a larger batch amortizes the copy. At
2048-token batches patch 108 hides nearly all of it (367 tok/s against 376
with no copies at all), so what is left there is compute: the expert matmuls
on mmq's MUL_MAT_ID (emulated int8 at ~2.4 effective TOPS), the dense
layers, attention and scheduling. No per-part breakdown is recorded, and
recording one is the next step: profile pp2048 with `HSA_ENABLE_SDMA=0`,
because the same profiling run with SDMA on hung the GPU (sdma0 ring
timeout). Kernel times carry over to the SDMA build; copy times do not.

**Options.**
- Done: patch 108 copies the next MoE layer's experts into a second device
  slot on a second stream while the current layer computes, with
  `HSA_ENABLE_SDMA=1` so the copy runs on the DMA engine (FINDINGS section
  9). pp2048 at 512 / 1024 / 2048-token batches: 193 / 250 / 295 -> 243 /
  321 / 367 tok/s, against a 376 tok/s bound with no copies at all. At
  512-token batches the copy is still longer than the compute it hides
  behind (12 GB/s over PCIe).
- Fused dequantization inside the SGEMM's operand load is unmeasured. Moving
  the experts onto patch 106's route instead (the next item) was measured and
  lost.
- Expert matmuls through dequantization + fp32 SGEMM (patch 106's route,
  forced for every expert slice through llama.cpp's per-expert MUL_MAT_ID
  path, dev-only switch): slower, pp2048 with experts in system RAM 292 ->
  223 tok/s at 2048-token batches and 167 -> 122 at 512 (SDMA off in both).
  That path synchronizes with the host to sort tokens by expert and then runs
  a dequantization and a GEMM per expert, 256 of each per layer. A grouped
  kernel that reads the quantized experts directly is the remaining form of
  this lever (unbuilt).
- MUL_MAT_ID tile choice for gfx803: measured, not a lever. Forcing every
  tile width from 8 to 64 for the expert matmuls (dev-only override) left
  pp2048 within noise: 171-176 tok/s at 1024-token batches for every width,
  238-257 at 2048.
- The PCIe link speed matters as much as any kernel. amdgpu's PCIe DPM can
  stay at its low level (2.5 GT/s x8) under load: `pp_dpm_pcie` then shows
  level 0 throughout a copy, a 512 MiB host-to-device copy runs at 3.4
  instead of 12.3 GB/s, and the same build's pp2048 (2048-token batches)
  falls from 367 to 247 tok/s. Generation barely changes. With the BIOS link
  setting on Auto the driver raises the link to 8.0 GT/s x16 for copy-heavy work
  (pp2048 at 2048-token batches then measures 364 tok/s) and leaves it at 2.5
  GT/s during training, which moves little data over the bus. Check
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

### RT-1: the cost of the vector-L1 invalidate

**Why.** The rebuilt `libamdhip64` publishes a PM4 `ACQUIRE_MEM` packet in an
AQL ring slot of its own before every kernel dispatch, which fixes the stale
vector L1 behind several silent wrong answers (FINDINGS section 4).
Back-to-back empty kernels take 4.3 us each with it and 2.65 us without, in
streams and in replayed HIP graphs alike, and a 4096-block kernel pays the
same 1.5 us. That is ~2.4 ms of an MoE token (~1,450 dispatches, ~6%; LLM-8
shows the drop), 2.3-2.7% of dense decode and ~2.4 ms of an fp16 SD 1.5 step
(~1,430 dispatches, under 1%). FINDINGS measured the L1 alone, L1 plus scalar
cache, and the L2 write-back plus invalidate pair at about the same decode
cost, so the cost is the extra packet, not the cache work.

**Options.** Carrying the invalidate in a packet the dispatch already
publishes instead of a slot of its own, if the command processor then
handles it faster: unmeasured. Fewer dispatches: every fusion in LLM-5 and
LLM-8 saves 4.3 us, not 2.65. Skipping the invalidate for some dispatches is
not an option without proof that their inputs cannot be stale; the bug it
fixes returns wrong numbers without an error.

**Verify.** FINDINGS section 4's table (the rocBLAS probe's 511 cases, the
MIOpen reduce shapes, the small sgemm loops, `SOLVE_TRI`) with the change,
and the empty-kernel timing.

### LLM-5: dense decode above the bandwidth bound

**Why.** Qwen3 8B Q4_K_M (4.86 GiB of weights besides the embedding table)
generates 27.81 +- 0.09 tok/s at a 5120-token context depth (35.96 ms per
token) and 31.16 +- 0.09 tok/s (32.09 ms) at zero depth. The 5120-token
context adds 3.87 ms per token. Reading the weights and the q8_0 KV cache at
the ~225 GB/s that q8_0 and f16 single-token matmuls reach takes 23.4 ms
(21.6 ms for 4.87 GB of weights, 1.8 ms for 0.40 GB of cache), ~12.5 ms less
than the wall time. The figures in this section predate the stale-L1 fix
(about 1.65 us more per dispatch) and patch 111 (LLM-9).

**Bandwidth measurements.**
- **Weights:** 4.87 GB per token (the 5.218 GB model file minus the 0.35 GB
  Q4_K token embedding table, of which one row is read). `mul_mat_vec_q`
  takes 24.54 ms per token (17.76 ms Q4_K across 162 launches, 6.77 ms Q6_K
  across 19 launches), ~198 GB/s: 2.9 ms above 225 GB/s.
- **KV cache:** 36 layers x 8 KV heads x 128 channels x 2 tensors x 5120
  tokens x 1.0625 bytes = 0.40 GB, 1.8 ms at 225 GB/s. Flash-attention
  decode takes 4.42 ms per token (4.19 ms `flash_attn_gcn4_decode_q8_0`,
  patch 102's kernel, across 36 launches; 0.23 ms
  `flash_attn_combine_results` across 36), ~90 GB/s effective, 2.6 ms above.

**Where the time above the bound goes.** Without HIP graphs under profiling,
wall time is 37.16 ms per token (26.91 tok/s), kernel execution sums to
34.07 ms, and idle between kernels is 3.09 ms (8.3%) across 834 launches per
token:

| Component | Time/token | Launches/token | Mean | Role |
| --- | --- | --- | --- | --- |
| `flash_attn` decode above the KV read | 2.6 ms | 36 | 116.3 us | ~90 GB/s effective against ~225. |
| `mul_mat_vec_q` above the weight read | 2.9 ms | 181 | 135.6 us | ~198 GB/s against ~225; without integer dot instructions the K-quant dot products are ALU sequences. |
| `mul_mat_vec_f` (attn_v) | 2.38 ms | 36 | 66.0 us | Unquantized F16 `blk.*.attn_v.weight` [4096 x 1024] in all 36 layers (8.4 MB/layer, 302 MB/token) at ~127 GB/s; 1.34 ms at 225 GB/s. |
| Small kernels (<10 us mean) | 2.97 ms | 581 | 5.1 us | Fixed launch and tail latency across 6 small-kernel families. |
| Residual idle between kernels | ~1.9 ms | - | - | 3.09 ms idle without graphs, reduced by 1.2 ms when HIP graphs are enabled. |

The small-kernel total (2.97 ms across 581 launches) breaks down into:
- `rms_norm_f32`: 73 launches, 0.65 ms (mean 8.9 us)
- `quantize_q8_1`: 181 launches, 0.65 ms (mean 3.6 us); patch 111 now
  converts an input shared by several products once
- `rms_norm_mul_rope_f32`: 72 launches, 0.55 ms (mean 7.7 us)
- `k_set_rows_quant`: 72 launches, 0.52 ms (mean 7.2 us)
- `fwht_cuda`: 144 launches, 0.35 ms (mean 2.4 us)
- `flash_attn_combine_results`: 36 launches, 0.23 ms (mean 6.5 us)

KV Hadamard rotation (144 `fwht_cuda` launches rotating Q, K, V, and attention
output) costs ~0.2 ms per token: generation measures 27.58 tok/s with rotation
enabled versus 27.75 tok/s with `LLAMA_ATTN_ROT_DISABLE=1`.

**Levers and expected gain** (estimates, unmeasured; the GPU part of an MoE
token runs the same kernels, LLM-8).
- **K-quant decode kernels:** patches 103 and 104 took them from 127-165 to
  ~198 GB/s; the rest to ~225 GB/s is ~2.9 ms per token.
- **Flash-attention decode** (patch 102's kernel): 2.6 ms above its KV read at
  5k depth, more at longer contexts, the case of a coding agent's prompts.
- **`mul_mat_vec_f` on small matrices:** 127 GB/s on `attn_v` against 222-229
  GB/s on large f16 matmuls, ~1 ms per token; the same kernel runs an MoE
  model's fp32 router. Requantizing `attn_v` to q8_0 (~160 MB) would save
  ~1.5 ms instead, but it changes the model file and its numbers, which is
  the model's choice and not a kernel fix.
- **Kernel fusion:** each launch costs at least 4.3 us (the card table). The
  candidates are `rms_norm` with `set_rows` (217 launches, 1.72 ms), `fwht`
  with rope/set_rows (144 launches, 0.35 ms) and the split-K combine at this
  depth (36 launches, 0.23 ms); fusing about a third of the small-kernel time
  saves ~1 ms.

### LLM-8: MoE generation with experts in system RAM

**Where the time goes.** Qwen3.6 35B-A3B Q4_K_M, all experts in host memory
(`-ncmoe 99 -lm none`), llama-bench tg64: 25.58 +- 0.95 tok/s, 39.1 ms per token.
The figures in this section were measured before the stale-L1 fix and before
patches 111 and 112. The same benchmark (tg128) measured 24.0-24.2 tok/s with
the fix, a difference that matches 1.65 us on each of the ~1,450 dispatches of
a token (~2.4 ms, RT-1); 24.5-24.6 with 111 added; and 25.2-25.7 with 112
(SDMA off and on).

| Part | Per token | How it was measured |
| --- | --- | --- |
| CPU expert matmuls | 17.2 ms | GGML_SCHEDTIME split timer (568.0 ms over 33 tokens); 0.56 GB of Q4_K expert weights per token is ~33 GB/s, reaching ~80% of the practical DDR4-2800 read ceiling (39-41 GB/s). At most ~3.5 ms/token is recoverable on the CPU side. |
| GPU splits (attention, DeltaNet, router, shared expert, output head) | 16.22 ms | Pure GPU kernel execution (1454 launches per token under rocprof, 38.2% of wall time). The split timer reports 27.0 ms for ROCm0 compute because it synchronizes before and after each split, including queue and barrier wait. |
| Hand-offs and idle gaps | 25.3 ms | GPU idle time (59.7% of wall time), dominated by 40.3 gaps of 0.1-1 ms totaling 17.60 ms (avg 0.44 ms per layer), where the GPU waits for the CPU expert split. Input H2D copies average 13.7 us across 41 calls (0.56 ms). |

What does not change it: CPU threads (4: 21.6, 8: 23.7, 12: 23.0, 16: 18.6
tok/s), the thread pool's polling (`--poll` 0/50/100), HIP graphs (on: 23.4,
off: 24.0; they are captured once per split and replayed, 2523 replays to 71
captures), `ROC_ACTIVE_WAIT_TIMEOUT` (100 to 10000 us), and disabling transparent
huge pages. Transparent huge pages for the expert buffer (madvise) are
untested.

**Synchronizations.** Before patch 112, which makes the next split wait on an
event instead of the whole stream, the workload issued 183
`hipStreamSynchronize` calls per token (20,000 recorded synchronizations over
109 tokens; over 33 tokens under tracing, HIP API calls include 6046
`hipStreamSynchronize`, 1723 `hipEventSynchronize` and 6009
`hipMemcpyAsync`). Total synchronization wait time is 19.7 ms per token,
attributed as follows:

| Call site | Calls/token | Mean wait | Time/token | Role |
| --- | --- | --- | --- | --- |
| `ggml_backend_cuda_synchronize` after `tensor_get_async` | 39.9 | 407.1 us | 16.23 ms | CPU waiting for preceding GPU split output to consume in CPU experts (needed data). |
| `ggml_backend_sched_synchronize` at token end | 1.0 | 2296 us | 2.30 ms | Waiting for logits D2H transfer (needed data). |
| Buffer set tensor path (`tensor_copy` fallback) | 41.9 | 23.1 us | 0.97 ms | Synchronous H2D copy of CPU split results back to GPU. Candidate for async stream ordering. |
| User input tensors (tokens, positions, masks) | 7.0 | 16.4 us | 0.115 ms | Small host-to-device inputs. |
| Redundant stream synchronizations (4 call sites) | 93.8 | 0.3-1.1 us | 0.06 ms | Idle-stream syncs: 42.9 at 1.09 us, 42.9 at 0.30 us, 7.0 at 0.42 us, 1.0 at 0.9 us. |

Of the 19.7 ms synchronization wait, 18.5 ms is the CPU waiting for GPU compute
to complete. Removing every redundant synchronization recovers only 0.06 ms
per token (0.15% of token time).

**GPU kernels by shape.** Under tracing without graphs, kernel execution
per token is:
- `mul_mat_vec_q`: 8.53 ms across 271 launches
- `mul_mat_vec_f`: 1.07 ms
- `rms_norm`: 0.97 ms across 191 launches
- `gated_delta_net`: 0.86 ms
- `quantize_q8_1`: 0.83 ms across 271 launches
- `get_rows`: 0.65 ms
- `bin_bcast`: 0.60 ms across 140 launches
- `topk_moe`: 0.57 ms
- `flash_attn` decode: 0.41 ms
- `cpy`: 0.29 ms

Large weight matrices run near memory bandwidth: the output head (248320 x 2048
q6_K) takes 2.09 ms (~200 GB/s) and 8192 x 2048 takes 63 us (150-220 GB/s).
Small shared-expert matrices do not: 2048 x 512 (down) takes 22 us (~27 GB/s)
and 512 x 2048 (gate, up) takes 15 us (~40 GB/s). Small-matrix variants (more
rows per workgroup) or combining gate and up could recover an estimated
1.5-2 ms per token (3-5%); the fp32 router (256 x 2048 at 21 us) about 0.5 ms
more.

**Levers.** Removing redundant synchronizations is not a lever (~0.06 ms/token).
Converting the synchronous CPU-to-GPU return copy (0.97 ms) to asynchronous
stream ordering bounds recoverable time to 0.97 ms (estimate ~0.5 ms). The token
decomposes into ~17 ms GPU compute, ~17 ms CPU compute, and ~8 ms serial
hand-off. Within a layer only the shared expert is independent of the routed
experts: attention and the router feed them, and the next layer needs their
sum. Patch 112 (FINDINGS section 9) runs the shared expert while the CPU
computes the routed experts, which reaches the bound measured by leaving the
shared expert out (tg128 24.6 -> 25.6 tok/s with SDMA, Ollama 25.3 -> 26.2).
Overlap across tokens is not possible in single-sequence generation. The
shared expert's small matrix-vector products above now run behind the CPU
split (about 80 us against 440 us per layer), so their speed no longer
counts; the fp32 router (~0.5 ms per token) and attention still do. What is
left is that kernel time, the hand-offs, and the CPU experts' own speed (at
most ~3.5 ms). The GPU split's kernels are LLM-5's (mmvq, rms_norm,
quantize_q8_1, `mul_mat_vec_f` for the router), so its levers apply here too,
and each of the ~1,450 dispatches pays the invalidate of RT-1. The CPU experts
are llama.cpp's generic CPU kernels, not gfx803 code.

### SD-10: fp16 GroupNorm/SiLU and GEGLU in the node

**Why.** The node's GroupNorm (with SiLU) and GEGLU kernels take fp32 inputs
only (`tools/gfx803_norm.py` and `tools/gfx803_geglu.py` fall back to PyTorch
otherwise), and ComfyUI runs the UNet in fp16 on this card, so its default
path fuses only the attention. GroupNorm, LayerNorm and SiLU are 10 ms (3%)
of the fp16 step, and GEGLU's split, GELU and product are part of the ~38 ms
of elementwise kernels, norms and casts. At fp32 the same kernels measured
1.0-1.35x (GroupNorm) and 1.6-1.7x (GEGLU) against PyTorch.

**Approach.** fp16 loads and stores with fp32 arithmetic in both kernels (the
norm keeps its float64 accumulation). Expected: a few ms per step, plus
fewer dispatches; unmeasured.

**Verify.** Against CPU float64 with fp16 inputs over the SD 1.5 and SDXL
shapes, and a ComfyUI image against the previous build.

### SD-11: ComfyUI time outside the UNet

**Why.** A warm 20-step SD 1.5 image at 512 px takes 8.5 s with the node, of
which the 20 UNet steps are 6.8 s; without the node 10.5 s, of which 7.8 s.
The other 1.7 or 2.7 s are the text encoder, the VAE decode, the model moves
of `--lowvram` (the comfyui image's default), and ComfyUI's own work, in
unknown shares. Why the remainder is 1 s shorter with the node is not known
either.

**Measure.** Time the text encoder, the sampler and the VAE decode
separately (the in-process harness in "Measuring"), and compare `--lowvram`
with `--normalvram`: SD 1.5's fp16 UNet is 1.7 GB and fits beside its VAE and
text encoder on 8 GB. SDXL's (~5 GB) is a separate measurement. The service's
flags are the user's choice; this target measures what they cost.

### SD-12: SDXL VAE decode

**Why.** Decoding a 1024 px SDXL latent takes 12.2 s at fp32 and 24.6 s at
bf16 when tiled, against ~55 s for the 20 UNet steps. With `--bf16-vae` the
convolutions run on im2col GEMMs (there is no bf16 Winograd and no bf16
shim tile), and ComfyUI falls back to tiled decoding when the regular decode
runs out of memory, on some images and not others (SDXL inference above).

**Measure.** The decode's time by convolution shape and solver
(`MIOPEN_ENABLE_LOGGING_CMD=1`), with and without tiling, at bf16, fp16 and
fp32, and its peak VRAM. Then the options: fp16 storage on the fp16
Winograd solver and shim tiles (if the VAE's activations stay in fp16 range,
which SDXL's VAE is known not to do everywhere), or a tiled fp32 decode with
less overlap.

### SD-2: Winograd F(4x4, 3x3)

F(4x4, 3x3) needs 2.25 multiplies per output against F(2x2, 3x3)'s 4, so the
GEMM part of an fp16 SD 1.5 step (~100 ms of the convolutions' 127) would
shrink by ~44%. Per output the transformed input and output shrink as well,
36 values per 16 outputs against 16 per 4; only the filters grow, to 6x6
tiles from 4x4 (2.25x), and the filter transform runs on every call (17 ms
per step now, ~38 ms at 6x6). Estimate for an fp16 SD 1.5 step: GEMM ~100 ->
~57 ms, input and output transforms 9 -> ~5 ms, filter transform 17 ->
~38 ms, about 25 ms (7%) saved; more at SDXL, where the GEMM's share is
larger. Two unknowns can shrink it. The GEMMs get a quarter as many rows
(512 at the 64x64 latent with CFG, 8 at 8x8), which runs less efficiently,
so the small latents may stay on F(2x2). And the transform coefficients
(up to 5 in the input transform, 8 in the output) amplify rounding, most of
all with fp16 storage of the transformed tiles.

MIOpen's own F(4x4) solvers (`ConvMPBidirectWinograd<4, 3>`) run gfx9
assembly transforms, so the route is extending this repo's
`ConvHipWinogradF2x3` (`winograd-f2x3-hip-gfx803.patch`) to 6x6 tiles. Before
touching MIOpen: time the shim's GEMMs at F(4x4)'s shapes, time the three
transforms in a standalone kernel, and measure the error against CPU float64
on every SD 1.5 conv shape at fp16 and fp32. Independent of SD-7.

### TRAIN-3: where a training step goes

**Why.** The fp16 step is 2.47 s (512 px, 4 accumulated images, gradient
checkpointing). SD-8 took attention out of the picture, and nothing records
where the rest goes: the forward, the checkpointed recompute, the backward,
AdamW, data loading. The fp16 NT GEMMs of the backward (26% of the GEMM
FLOPs at 384 px) run on the older 16x16-thread tiles: the one- and
two-wavefront tiles of BLAS-7 serve fp16 batched NN and TN only.

**Measure.** CUDA events around the phases of a steady step, and the shim's
call log (`GFX803_SGEMM_SHIM_VERBOSE`) for one step's GEMMs by layout and
shape, timed at sustained clocks.

### BLAS-7: shim GEMM efficiency at SD shapes

**Why.** The shim's GEMMs were 202 ms (57%) of an fp16 SD 1.5 step before the
two-wavefront TN tile: about 917 GFLOP at ~4.5 TFLOPS on average, about two
thirds of the card's FMA peak and three quarters of the ~6.2 TFLOPS it
sustains under its power limit.
Each further 0.5 TFLOPS on average is about 25 ms per step, and training's
GEMMs gain as well.

**Where.** `rocm6.4.4/patches/sgemm-shim-batched/gfx803_sgemm.h`. The step's
GEMMs, from the shim's call log (`GFX803_SGEMM_SHIM_VERBOSE` over the
profiled steps only, so MIOpen's Find calls are excluded): TN linears
(m = 320/640/1280, n = 154 to 8192, k = 320 to 5120), the Winograd solver's
strided-batched NN GEMMs (batch 16, m = 32 to 2048, n = k = 320 to 1280), and
1x1 convolutions as batch-2 NN GEMMs (m = 64 to 4096).

**Done.** Two dispatch rules had been tuned on fp32 operands only. Single-
float B staging for TN with K >= 2048 lost to float4 B at fp16 (SD 1.5's
feed-forward output projections, 1280x512x5120: 2.1 -> 3.5 TFLOPS;
640x2048x2560: 2.8 -> 4.3), and the 64x64 tile lost to the 32x32 tile at
fp16 for M = 32 (the Winograd GEMMs at the smallest latent:
16 x 32x1280x1280 0.86 -> 0.63 ms, k = 2560 2.33 -> 1.46 ms). Both now apply
to fp16 only; every tile and staging accumulates in the same order, so
outputs are bit-identical. fp16 SD 1.5 step 415 -> 399 ms.

The fp16 tiles widened each operand to fp32 right after its global load. The
compiler then waits for that load (`s_waitcnt vmcnt(0)`) inside the fetch,
one load at a time, before the slice's multiply-adds start, so every K slice
paid four or five memory round trips that nothing overlapped. The values now
stay fp16 in registers until the next slice's LDS store, and the loads stay
in flight during the multiply-adds as they do at fp32. A one-wavefront tile
(`gfx803_w_kernel`: 64x64, an 8x8 block per thread, no barriers) takes the
fp16 strided-batched NN calls with batch >= 8 and M >= 512, the Winograd
solver's GEMMs at the larger latents (SDXL 8192x320x640 b16: 13.6 -> 9.8 ms).
Outputs are bit-identical to the previous kernels on 478 cases (all layouts,
odd sizes, padded leading dimensions, offset pointers, batches, the SD
shapes). GEMM time over the shim call logs, fp16: SD 1.5 707 -> 620 ms,
SDXL 2426 -> 2120 ms; fp32 unchanged. fp16 UNet step: SD 1.5 392 -> 360 ms
(457 -> 395 ms without the node, whose split attention also runs batched
GEMMs), SDXL 3.34 -> 2.98 s.

A two-wavefront tile (`gfx803_w2_kernel`: 64x128, 128 threads, each
wavefront a 64x64 half with 8x8 per thread, sharing the A piece) takes fp16
TN calls with whole tiles and at least three rounds of tiles over the CUs
(M/64 * N/128 * batch >= 3 * 4 * CUs). In TN both operands are contiguous in
K, and the one-wavefront tile's 8-deep pieces read 16 bytes of each 64-byte
line, so L2 delivers each line up to four times; a wider fetch in one
wavefront does not fit (two pieces at a time already take 127 VGPRs). The
two wavefronts share a 16-deep slice's fetch, hold the next slice in 12
VGPRs while multiplying the current one out of one 12 KiB LDS buffer, and
synchronize twice per slice; each barrier's drain of outstanding loads
comes after a whole slice of multiply-adds. SDXL's large linears, timed at
sustained clocks: TN 10240x2048x1280 12.4 -> 10.2 ms (4.3 -> 5.3 TFLOPS);
in short timings at the boost clock 5120x8192x640 13.9 -> 10.1 ms and
2560x8192x320 2.84 -> 2.37 ms. Bit-identical on the 478 cases with
the tile forced on every call it supports, and on 25 more with K a multiple
of 16 but not 32, padded leading dimensions, offsets and batches. fp16 UNet
step: SDXL 2.98 -> 2.84 s, SD 1.5 359 -> 355 ms.

SDXL's 1280-wide linears have too few 64x128 tiles for the rule (320 for
1280x2048) and stay on the 64x64 tile, at 4.1-4.6 TFLOPS sustained:
1280x2048x1280 1.45 ms (372 calls per step) and the feed-forward output
projection 1280x2048x5120 6.5 ms (60 calls). The second also shows why
the two-wavefront tile is limited to many tiles. Its rows are 10240 bytes
apart, a multiple of 2048, so the 192 rows a tile reads at one k fall in
the same DRAM channel, and the two-wavefront tile, which reads half of
each 64-byte line per slice and needs L2 to keep the other half, took
9.2 ms. Every row stride that is a multiple of 2048 bytes does the same
(K = 3072, 4096, 6144 at fp16); with rows padded by 64 halves it ran 4.9-6.4
ms, depending on where the buffers landed. At 2048 tiles (4096x4096xK) it
stays at or ahead of the 64x64 tile even at those strides, so the rule
needs no stride test. Padding the operands into a copy would take scratch
memory, which the shim has none of by design, for at most ~80 ms of
SDXL's step.

**Left, by time lost per step against 4.8 TFLOPS** (each shape timed alone):
TN with n = 128/154 (cross-attention K/V projections and the 16x16 level)
at 1.0-4.3 TFLOPS (~3 ms), the batch-2 NN 1x1 convolutions with m = 64-256
at 2.0-4.6 TFLOPS (~1.3 ms), and the 30 TN 1280x512x1280 linears per step at
4.3 TFLOPS (~1.2 ms). None alone is more than 1% of the step; a broader gain
needs a faster main tile rather than more dispatch rules. The two-wavefront
tile loses where a call has fewer than three rounds of its tiles (TN
1280x512x1280 0.41 -> 0.51 ms, 640x2048x640 0.38 -> 0.44, and
1280x2048x5120 6.5 -> 9.2 ms, where the channel camping above adds to it),
and the rule leaves those calls on the 64x64 tile. A 64x64 two-wavefront
variant for them is what is left of the TN linears.

**Measured, not adopted** (short timings at the boost clock; the ratios
hold). Ablating a 64x64 4x4-per-thread NN tile (2048^3, fp32) attributes its
time to the global fetch more than to LDS: removing the
fetch 3.96 -> 5.06 TFLOPS, removing the LDS reads 3.96 -> 4.52, removing the
barrier 3.96 -> 4.25, a pure multiply-add loop 6.35. 8x4 and 8x8 blocks per
thread inside the 16x16-thread workgroup did not help at SD sizes (fewer
workgroups, one wave per SIMD); 32-deep K slices halved the speed (LDS limits
a CU to one workgroup). Fetching two slices ahead with LDS double-buffered
gains 5-20% on single fp16 GEMMs (2048x640x640 3.1 -> 3.6 TFLOPS) but loses
5-15% on the batched Winograd and 1x1-convolution GEMMs that make up the
step's NN time (116 -> 126 ms per step) and on fp32.

The one-wavefront tile, run on every call the 64x64 tile serves, loses on
single GEMMs: SDXL's fp16 GEMM time 2426 -> 3513 ms, TN 1280x2048x5120 6.6 ->
21.5 ms per call. On TN 1280x2048x1280 it reaches 3.3 TFLOPS against the old
tile's 4.5; removing its global fetch takes it to 5.6 and removing its LDS
stores to 5.7, while removing its LDS reads changes nothing. With no second
wavefront in the workgroup, only other tiles on the same SIMD can hide its
loads, which a batched call supplies and a single GEMM at SD sizes does not.
Fetching two K rows per load (`GFX803_SGEMM_WAVE_KC2=1`) recovers part of it
(TN 1280x2048x5120 12.3 ms) but not enough; 64x32 tiles and tile grouping did
not change the picture. At fp32 it runs batched NN at 0.5-2.8x the old time
with no shape pattern, so fp32 keeps the old tiles.

### SD-9: SDXL attention

SDXL spends 0.42 s of its 2.75 s step in the fused attention, 0.39 s of it
in self-attention at 4096 tokens (10 calls) and 1024 tokens (60 calls),
head size 64. Sustained, per CFG-batch-2 call: 4096 tokens 21.2 ms, 1024
tokens 2.68 ms, about 4.0 TFLOPS; SD 1.5's head size 40 at 4096 tokens
12.5 ms (3.4 TFLOPS), head size 80 at 1024 tokens 1.43 ms (3.8).

The kernel runs one wave per SIMD (one block per CU: 51 KiB of LDS at head
size 64 and about 200 VGPRs), and with nothing to co-issue with, every
instruction a wave issues besides its multiply-adds takes a slot from them.
Its main loop is 91% `v_mac_f32`, so the multiply-adds alone would allow
~5.4 TFLOPS at the sustained clock. Timing builds with one cost removed at a
time, before the change below: Q*K^T's LDS reads 0%, the 128 `ds_bpermute`
per tile that fetched P for P*V 10%, the V reads with them 11%, the
exponentials 4%, the K/V tile loads, stores and barriers 8%.

**Done: P through DPP.** The lanes now hold the score tile so that a query
row's 32 scores lie in one 16-lane DPP row, and each step of P*V takes the
key group whose P values are a fixed rotation away, which `v_mac_f32_dpp`
applies to its operand for free (inline asm: the compiler emits a separate
`v_mov_b32_dpp` for `__builtin_amdgcn_mov_dpp`, and with those 112 moves
per tile the kernel was 12% slower than with the bpermutes). K and V rows
in LDS got a D+4 stride like Q: in this layout the eight key groups of a
16-lane row read eight K rows at once, which unpadded share one bank group
(9% slower). The softmax's xor-2 lane exchange is a DPP quad permutation.
SDXL step attention 504 -> 416 ms (step 2.84 -> 2.75 s), SD 1.5 91 -> 79 ms
(step 355 -> 342 ms). Q*K^T, the row maxima and the row sums combine the
same values in the same order as before, so the log-sum-exp the training
backward uses is bit-identical; P*V sums its keys in a different order per
lane, and the fp16 outputs differ from the previous kernel's by at most
one fp16 rounding step, at the same distance from a float64 reference.

**Left.** A second wave per SIMD would hide the remaining LDS reads,
waits and barriers, but needs 128 VGPRs and half the LDS (the fp32 Q tile
alone is 34 KiB). The training backward (SD-8) takes P and dS for dV/dK
the same way.

### LLM-10: image prompts

**Why.** A 1080p image becomes about 2000 tokens and takes 56 s with
Qwen3.5 9B, where 2000 text tokens take about 15 s. Where the time goes is
unmeasured: the vision encoder (the mmproj model, run by llama.cpp's
multimodal code), its placement (patch 090 fixes where its memory is
reserved), and the image tokens' own prompt processing.

**Measure.** The encoder's time and the prompt's time separately (llama.cpp's
multimodal timings), then the encoder's kernels under rocprof on the patched
runtime.

### SD-7: fused Winograd F(2x2, 3x3)

**Why.** Convolutions are 127 ms (37%) of an fp16 SD 1.5 step. At fp16 the
input and output transforms take 9 ms per step and the filter transform
17 ms; the rest (~100 ms) is the GEMM, which a fused kernel would still have
to run. At fp32, before the shim's fp16 tiles, ~85 ms of 182 was outside the
GEMM. `ConvHipWinogradF2x3`
(`rocm6.4.4/patches/miopen/winograd-f2x3-hip-gfx803.patch`) transforms the
input into 4x4 tiles (4x the data of the input) in one kernel, writes them to
DRAM, runs a strided-batched GEMM over them, writes the result, and
transforms it back in a third kernel. Those round trips through DRAM are most
of the time outside the GEMM.

**Approach.** One kernel per output tile block that transforms input tiles in
LDS, runs the 16 small GEMMs from LDS and registers, and transforms the
output before writing it. This is roughly what MIOpen's binary Winograd
kernels (`ConvBinWinogradRxS`) do as persistent kernels, and those are
disabled on gfx803 because they corrupt memory and hang the GPU (FINDINGS
section 2), so build it as ordinary HIP with ordinary launches. Forward and
backward-data both matter (training).

**Expected gain.** At fp16 at most the 9 ms of input and output transforms
per step (2%), plus the GEMM's own reads and writes of the transformed
tiles; at fp32 and in training more, unmeasured. Measure the split
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

### SD-1: Winograd weight gradients

`ConvHipWinogradF2x3` covers forward and backward-data. LoRA training freezes
the convolution weights and never needs backward-weights; full fine-tuning
or LoCon would, and falls back to MIOpen's GEMM path (im2col, then GEMM). The
weight-gradient form is an input transform, an output-gradient transform, a
batched GEMM accumulated over tiles, then the filter transform.

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

- **DPP for reductions.** Reductions are a small share of every workload
  measured here; DPP pays where it moves operands into multiply-adds (SD-8,
  SD-9).
- **Transforming the Winograd filters once instead of per call.** The 48
  Winograd convolutions of an fp16 SD 1.5 step read 1.1 GB of 3x3 weights and
  write 1.9 GB of 4x4 tiles: ~13.4 ms at the streaming rate against 17.1 ms
  measured, so the kernel is near its bound. Keeping the transformed filters
  would cost 1.9 GB of VRAM beside 1.7 GB of fp16 UNet weights on an 8 GB
  card.
- **HIP graphs for decode.** Already on by default and in use on gfx803: they
  save ~1.2 ms of idle per dense token (LLM-5), and MoE generation measured
  the same within noise either way (LLM-8).
- **Routing generic fp16 GEMMs in every layout to the unpipelined shim.**
  Slower than rocBLAS (633.5 -> 675.4 ms per fp16 SD 1.5 step); the pipelined
  fast tiles with fp16 operands (BLAS-4) are what made the fp16 route pay.
- **Redefining `C10_WARP_SIZE` for gfx8** (PT-1 below): a global macro change
  breaks kernels written for the old value.
- **LoRA `addmm` with beta=1 fusion.** Accounts for ~360 launches of ~10 us each
  (~4 ms of a 2.5-3.4 s training step, under 0.2%), which does not justify a
  patch.

## Figures to re-measure

Figures this file or FINDINGS still quotes that later changes may have moved,
and claims that wait for a run. Most need the card; several can share one
session.

| Figure | Where | Why it may have moved | How |
| --- | --- | --- | --- |
| Dense decode breakdown (Qwen3 8B 27.81 tok/s, LLM-5's table) | llama.cpp decode, LLM-5 | stale-L1 fix (+1.65 us per dispatch), patch 111 | rocprof on the patched runtime with the desktop stopped; llama-bench tg at 0 and 5k depth |
| MoE generation breakdown (25.58 tok/s, LLM-8's tables) | LLM-8 | stale-L1 fix, patches 111 and 112 | `GGML_SCHEDTIME` split timer and rocprof, `-ncmoe 99` |
| SD 1.5 and SDXL kernel breakdowns (GEMMs 202 ms and 2.23 s per step) | Where the time goes, BLAS-7 | two-wavefront TN tile, SD-9 | `torch.profiler` on the patched runtime |
| SDXL end to end and VAE (83 and 99 s, 67 s of sampling) | SDXL inference, SD-12 | UNet step 3.36 -> 2.75 s | ComfyUI, 20 steps, both VAE precisions |
| fp32 SD 1.5 image (35 -> 10.5 s; 644 and 526 ms per step) | front page, FINDINGS "Speed, start to finish" | SD-6 and SD-9 attention, stale-L1 fix | ComfyUI `--force-fp32`, warm images |
| The memory bandwidth bound | card table, LLM-5 | LLM-5's estimates take the 222-229 GB/s of q8_0 and f16 matmuls as the bound; no read-only streaming kernel has measured it at sustained clocks | a read-only streaming kernel at sustained clocks |
| `test-backend-ops`, full suite | Verifying a change | stale-L1 fix (FINDINGS section 4 has `SOLVE_TRI` passing with it), patches 108-112 | the full suite in the current Ollama image against the CPU backend |
| The va-reuse-defer runtime change and the small-shape rocBLAS patch | FINDINGS section 4 | the stale-L1 fix may explain the bugs they were written for | their original reproducers without them, with the fix |
| Kernel TFLOPS in BLAS-1 to BLAS-6 and BLAS-7's ablations | Done, BLAS-7 | short timings at the boost clock (10-20% high) | re-time only where a decision depends on the absolute value |

## Done

### TRAIN-2: fp16 training over a full-length run

One default `gfx803-train` run per precision, 1200 steps at 512 px on the
same dataset and seed, in the same image (fp16 53 min, peak 4.4 GB; fp32
62 min, 5.9 GB; neither faulted). The two precisions track each other step
for step: the paired losses correlate at
1.0000, fp16's is lower by 0.0007 on average (0.0001 over the first 100 steps,
0.0013 over the last), against a step-to-step spread of 0.10, and neither run
has a non-finite loss. The LoRA weights end as far apart (median per-tensor
cosine 0.968, relative difference 0.26) as one run's step-800 and step-1200
checkpoints are from each other, and images rendered with the two at 1200
steps show the same subject and style, for most prompt and seed pairs the same
composition. fp16 costs no measurable training quality.
`gfx803-train` keeps fp16 as its default.

### TRAIN-1: fp16 LoRA training

SD 1.5 LoRA at 512 px with `--mixed_precision=fp16` against fp32, steps 5-20
of a 20-step run with a warm find-db: 2.87 -> 2.47 s/step with gradient
checkpointing (card peak 5.8 -> 3.6 GB), 2.07 -> 1.73 s/step without (7.1 ->
4.9 GB). The gain is memory traffic: the fp16 GEMM tiles (BLAS-4, BLAS-7)
read half the bytes, and the arithmetic runs at the fp32 rate either way.
Without gradient checkpointing, fp16 fits beside a desktop. `gfx803-train`
trains in fp16 by default; `--precision fp32` remains. TRAIN-2 compares the two
over a full-length run.

### SD-8: fused attention backward

`tools/gfx803_fused_attention.hip` (`attn_bwd::attn_bwd_main`, `attn_bwd::attn_bwd_dq_reduce`, and an
LSE output from the forward), used by `gfx803-train` through `--sdpa`. A fp32
SD1.5 UNet forward and backward at 384 px spends ~97 of 387 ms in PyTorch's
attention at the 48x48 level and 1.4 GB on its matrices; at 512 px it runs
out of 8 GB without gradient checkpointing, and fits in 4.6 GB when the
attention keeps no N x M state.

Design: a block of 2x2 waves owns 64 keys (32 for head size 80), keeps K and V
in LDS and dK/dV in registers, and streams 64-row query tiles; S and dP are
computed once per tile pair from float4 LDS reads, P and dS reach the dV/dK
accumulators of other lanes through DPP row rotations inside the
multiply-adds (the lane tiling below) and the dQ ones through
`ds_bpermute`, LDS rows are
XOR-swizzled instead of padded (padding would push head size 64 past 64 KiB),
operands go through raw buffer loads (one 32-bit offset per access, rows past
N read 0), and dQ is written per key block to scratch slabs that a second
kernel adds in a fixed order. Per call at 4096 tokens, head size 40: 17.6 ms
at sustained clocks (4.3 TFLOPS-equivalent), against 43.2 ms for the previous two-kernel version
that computed S and dP twice, and 59 ms plus 2.6 GB for PyTorch; 2304 tokens
5.9 ms; head size 80 at 1024 tokens 2.9 ms (PyTorch 5.9). A two-kernel
version without scratch built the same way measured 23.0 ms. Training (fp32,
512 px, 4 accumulated, 40 steps, seed 42): 3.37 -> 2.87 s/step, 7.6 -> 5.6 GB;
2.03 s/step at 6.9 GB with `--no-checkpointing`; identical loss.

The lane tiling: lane l has row group (l/2)%8 and key group l%2 + 2*(l/16),
so the P and dS values one lane's dK/dV keys need from the 8 row groups lie
in its own 16-lane DPP row, at rotations of 2, 4, ... 14 lanes, which
`v_mac_f32_dpp` applies to its operand. That took 224 of the 336
`ds_bpermute` per tile pair out of the kernel. Sustained, per call:
head size 40 at 4096 tokens 18.8 -> 17.6 ms, head size 64 at 4096 34.4 ->
33.8 ms, head size 80 at 1024 3.02 -> 2.90 ms; gradients as close to a
float64 reference as before. Unrolling the eight rotation steps (the DPP
control is an immediate) let the compiler hoist each step's lane-dependent
addresses out of the query loop, which pushed head sizes 64 and 80 into
spilling inside it (80: 15% slower); an empty asm on the step's row index
keeps them in place, and the kernel stays at 231-245 VGPRs without spills.
The `dgroup` tables were searched again for the rotation order, and dQ got
one for head size 64.

Left: the slab scratch is 336 MB at 4096 tokens (537 MB at head size 64).
dQ's 112 remaining `ds_bpermute` per tile pair cost 2-6% (timing build
with them removed); a 16-lane DPP row holds 8 row groups but only 2 key
groups, so dQ's exchange across key groups cannot move to DPP alongside
dV/dK's.

### SD-6: multi-wave fused attention

Done in the node (`tools/gfx803_fused_attention.hip`): a block
of four waves handles 128 query rows and shares each 32-key K/V tile, P
values move between lanes instead of through an LDS tile (since SD-9 inside
the multiply-adds, by DPP), and the next K/V tile is prefetched into
registers. Per call, before SD-9: SD 1.5 64x64
self-attention (CFG batch 2, head size 40) 26.3 -> 14.3 ms, 32x32 (head
size 80) 1.95 -> 0.84 ms, SDXL 4096-token self-attention (head size 64,
batch 1) 34.8 -> 12.2 ms; SD 1.5 step attention 129 ms (ComfyUI's own) -> 91 ms.
The kernel reads Q, K and V in ComfyUI's [b, tokens, heads*D] fp16 layout
and writes the output there, which saves the transposes, fp32 casts and
output copy around every call (160 launches per step, 99 -> 92 ms); the K/V
prefetch keeps the fp16 values raw until they go to LDS, or the conversion
waits on every tile's loads (head size 40: 75.8 against 84.3 ms per step).

What made the difference was the multiply-add encoding, not the tiling: the
same multi-wave kernel written with `fmaf()` measured 15.3 ms at head size
40 and 29.8 ms at 64, because gfx803 issues the fused `v_fma_f32` at a
quarter rate (see the card table).

**Left.** Head size 160 stays on ComfyUI's split attention (0.34 ms per
call against the node's 0.79). Head sizes 40-80: SD-9.

**Shared with llama.cpp.** Patch 101's fp32 flash-attention tile kernel
(`fattn-tile-kernel.cuh`) already compiles its inner products to
`v_mad_f32`; the gfx803 build's `v_fma_f32` instructions are division
sequences. The quarter-rate FMA is not a lever there.

### PT-2: PyTorch's warp-size mixing

In gfx8 device code `C10_WARP_SIZE` is 32 (`c10/macros/Macros.h` gives 64 to
gfx9 only), while the hardware runs 64-lane wavefronts and host code's
`at::cuda::warp_size()` returns 64. A kernel that is launched or sized by
one and computes with the other is wrong on gfx803 and correct on every
architecture upstream tests. Four cases are fixed: layer_norm (PT-1); the
binary prefix scan behind `torch.topk`
([`gfx803-topk-scan-warp-size.patch`](../rocm6.4.4/patches/pytorch/gfx803-topk-scan-warp-size.patch)),
which wrote top-k results to wrong slots and, through its assert, hung the GPU
until the driver reset it; index_put with accumulate
([`gfx803-index-put-warp-size.patch`](../rocm6.4.4/patches/pytorch/gfx803-index-put-warp-size.patch)),
which dropped half of the repeated indices' sums; and embedding backward
([`gfx803-embedding-backward-row-width.patch`](../rocm6.4.4/patches/pytorch/gfx803-embedding-backward-row-width.patch)),
which returned wrong weight gradients and faulted on some shapes. group_norm shows the
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

`layer_norm_kernel.cu`, the two index_put launchers in `Indexing.cu` and
`embedding_backward_feature_kernel` in `Embedding.cu` are done. A read of all
14 files named nine suspect kernels; a GPU-against-CPU sweep (float64
reference, fp32 and fp16, shapes chosen to reach each kernel's path) found
only the embedding one wrong. Softmax and log-softmax forward and backward
(persistent and block paths), embedding renorm, embedding_bag's
per-sample-weights backward, depthwise conv2d's weight gradient and top-k's
blockwise counts all match the reference. A second pass over the other 18
files in `aten/` that use either value found five more suspects (top-k's
multi-block path, sparse `coalesce` with dense dimensions above 32, `sort` on
33 to 128 elements, `multinomial`'s row normalization, and the top-k scan the
patch above fixes); all match the reference on the GPU.

Mixing the two is not a bug by itself: a kernel whose shuffles use offsets
below 32 and whose shared memory is sized from `blockDim` treats each half of
a 64-lane wavefront as its own warp and stays correct (layer_norm's kernels
are like that once launched consistently). The bug is a disagreement inside
one kernel: a 64-lane ballot or lane-0 store counted with 32-lane warp
numbers (the top-k scan), a launch or shared-memory size from one value and
the kernel's indexing from the other (layer_norm), or a fixed block shape
that assumes 32 lanes per warp. The audit listed, for each file, every
kernel, the warp size its launch and shared memory use and the warp size its
device code assumes (`C10_WARP_SIZE`, `warpSize`, `__lane_id()`, ballots,
shuffle widths), and marked the disagreements.

Each suspect got a GPU-against-CPU test of its op, run headless with
`AMD_SERIALIZE_KERNEL=3` so that a trapping assert names its kernel and costs
one reset.

**Left.** PyTorch's own OpInfo tests (about 600 ops, against the suite's
118), run the same way, one op per process, would find the cases the audit
misses.

### LLM-9: one q8_1 conversion per shared input

[`111-mmvq-reuse-quantized-src1.patch`](../rocm6.4.4/patches/ollama/llama-cpp/111-mmvq-reuse-quantized-src1.patch):
each quantized matrix-vector product converted its fp32 input to q8_1 in a
kernel of its own, also when the product before it had just converted the
same input (attention's q, k and v; a DeltaNet layer's four input
projections), and every repeat cost a dispatch. The converted input is now
kept until a node writes over it. Generation: Qwen3 8B 30.1 -> 30.3-30.8
tok/s, Qwen3.6 35B-A3B with experts in system RAM 24.0-24.2 -> 24.5-24.6;
the same perplexity to the last digit. Not specific to gfx803.

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

### BLAS-4: fp16 operands on the fast tiles

rocBLAS's fp16 GEMMs in the TN, NN and TT layouts widen each fp16 operand to
fp32 inside the inner loop and ran 1.3-1.9x slower than the shim's fp32 tiles
at equal shapes (the NT layout was worse still and already on the shim). The
fast tiles now read fp16 operands with wide loads, widen once, accumulate in
fp32 and round once at the store, in all four layouts and the batched calls:
31 GEMM cases 1.0-2.4x faster (median 1.5x) at the same error against
float64, and the fp16 SD 1.5 step 602 -> 483 ms (convolutions 228 -> 147 ms,
linears 176 -> 137 ms). Details in FINDINGS section 8.

### SD-3, SD-4, SD-5: ComfyUI node

[`tools/ComfyUI-gfx803-attention/`](../tools/ComfyUI-gfx803-attention/)
replaces three operations in ComfyUI's SD 1.5 and SDXL UNets with HIP kernels from
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
  one pass; fp32 only, like the GroupNorm, so at ComfyUI's default fp16 only
  the attention runs (SD-10).

Correct against CPU float64 (attention within 3.5e-6 relative, GroupNorm and
GEGLU within 1e-5, over odd shapes, 2-D/3-D and non-contiguous inputs); a
full ComfyUI txt2img differs from a CPU run's by at most 0.3/255. An SD 1.5
fp32 step at 512 px: 644 -> 526 ms. Per call (batch 2, 8 heads), ms:

| | PyTorch SDPA | ComfyUI split | node |
| --- | --- | --- | --- |
| self-attention 64x64 (D 40) | 58.1 (2.3 GB extra) | 49.7 | 26.3 (10 MB) |
| cross-attention 64x64 | 2.20 | 1.09 | 0.69 |
| self-attention 32x32 (D 80) | 4.68 | 3.95 | 3.61 |
| self-attention 16x16 (D 160) | 0.49 | 0.34 | 0.79, not used |

GroupNorm 0.194 -> 0.166 ms at (2, 320, 64, 64); GEGLU 1.075 -> 0.638 ms at
8192 x 2560. The node is not part of the images (the comfyui image carries
only the fused attention, for training):
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
- **ComfyUI end-to-end timing.** The first image after a start with a fresh
  find-db runs MIOpen's Find (~2 minutes for SD 1.5 at 512 px); time the
  images after it, in the same container, and use several seeds.
- **Older kernel figures.** The TFLOPS figures in BLAS-1 to BLAS-6 and in
  BLAS-7's ablations are short timings that ran at the boost clock: compare
  them with each other, not with sustained figures, which read 10-20% lower.
- **SD harness.** ComfyUI can run in-process: put `/opt/ComfyUI` on
  `sys.path`, set `sys.argv` to ComfyUI's flags, call
  `comfy.options.enable_args_parsing()`, load with
  `nodes.CheckpointLoaderSimple().load_checkpoint(...)`, and use
  `model.model.diffusion_model` directly or `nodes.KSampler().sample(...)`.
  Use `--use-split-cross-attention` for anything with gradients (ComfyUI's
  default sub-quadratic attention writes `exp(..., out=...)` in place and
  cannot be differentiated). The service runs ComfyUI's default fp16 UNet with
`--lowvram --use-split-cross-attention`; add `--force-fp32` for fp32
comparisons. A 512 px
  forward and backward with PyTorch's attention does not fit in 8 GB without
  gradient checkpointing; with the fused training attention (SD-8) it does.
- **GEMM speed.** Time `A @ B` with CUDA events; build transposed operands
  as `.t()` views to reach each rocBLAS layout. Keep the card under load for
  about 2 s before the first timing and time each case for tens of
  milliseconds, so every number is taken at the power-limited ~1.34 GHz
  rather than the boost clock (see the card table); a variant timed first,
  or with a few calls, otherwise reads up to 20% fast. Buffers at different
  physical addresses also run at different speeds (DRAM channel mapping),
  and on some kernels the difference between two processes reaches 25%:
  alternate the variants over several processes and compare medians.
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
- **Node kernels.** `tools/ComfyUI-gfx803-attention/install.sh` recompiles a
  kernel library whenever its `.hip` source is newer than it, then copies the
  node into ComfyUI; restart ComfyUI afterwards. The comfyui image builds the
  training attention from `tools/` (`docker build --build-context
  tools=../tools ...`).
- **llama.cpp.** Patches in `rocm6.4.4/patches/ollama/llama-cpp/` apply with
  `git apply` in `Dockerfile-Ollama`, each checked by a marker grep. The
  scheduler patches in `llama-cpp-sched/` (108, 112) apply before the CPU
  build, because the scheduler is compiled into the `libggml-base` that build
  installs, and the build greps the installed library for their markers. To build
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
- **Training:** 20 `gfx803-train` steps (fp16, the default; also with
  `--precision fp32` when a change touches fp32 paths); avr_loss 0.15-0.2 is
  healthy, 3-4 means broken GPU math.
- **Node kernels:** `tools/correctness-suite/pytorch/attention_check.py` and
  `attention_train_check.py` (forward and backward against float64), then a
  ComfyUI image against the previous build.
- **llama.cpp:** `test-backend-ops test -o MUL_MAT`, `MUL_MAT_ID`,
  `FLASH_ATTN_EXT` (or the full suite, whose `SOLVE_TRI` cases above 64x64
  need the stale-L1 fix) against the CPU backend, and Ollama
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
- **Check that the new code runs before timing it.** A scheduler patch was
  measured as giving nothing while its condition never matched; a counter or
  a log line on the new path shows that at once.
- **Name the configuration in every saved result.** A log taken with a tile
  forced on every call (`GFX803_SGEMM_W2=1`) was later read as production
  timing and produced a target that did not exist.

