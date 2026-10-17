# Findings: what breaks on gfx803, why, and what this fork does about it

AMD removed gfx803 (Polaris: RX 470, 480, 570, 580, 590) from ROCm after 6.0.
The code still compiles for it, but much of the stack no longer produces
correct results, or runs far below what the hardware can do. This document
collects what this fork found while making SD 1.5 and SDXL LoRA training,
ComfyUI and Ollama work on a Polaris card, and what it did about each problem.

## How to read this document

**Audience.** People who debug GPU software: you are comfortable with
kernels, wavefronts, BLAS libraries and driver logs, but you may not know this
stack's internals. Sections give the background a finding needs and then the
evidence, so you can follow the reasoning and check it rather than take it on
trust. The [glossary](#glossary) covers the terms that recur.

**Shape of a finding.** Most sections follow one pattern: *symptom* (what you
see), *cause* (as far as it is known, and what was ruled out), *fix or
mitigation* (with the patch or file that holds it), and *check* (how to
reproduce the problem or confirm the fix, usually against a float64 CPU
reference). Where the cause is not known the section says so.

**Using it with an AI agent.** The document is written to be searched as well
as read.

- Start from the index below. Each row is one problem: its status, the section
  that explains it, and the file that holds the fix. Status words are fixed
  (see the legend), and a patch or tool is always named in backticks with its
  path, so `grep` finds both the finding and its code.
- Paths are relative to `rocm6.4.4/` unless they start with `tools/`, `docs/`
  or `../`.
- Treat a finding as a lead, not as a fact about your tree. Read the pinned
  source before trusting "broken" or "fixed": upstream may have changed the
  code, and a patch that applies cleanly proves nothing about correctness. The
  common failure on this hardware is a run that completes with wrong numbers,
  so every fix here was verified against a CPU reference, and so should yours
  be.
- Before writing a gfx803 patch for a bug, check whether another architecture
  with the same source also fails. If it does, the bug is upstream's, and a
  gfx803-only patch helps nobody else who hits it. Several rows below are
  marked "all GPUs" for that reason.
- Trace before you patch: `MIOPEN_ENABLE_LOGGING_CMD=1 MIOPEN_LOG_LEVEL=6`
  and `MIGRAPHX_TRACE_COMPILE=1` show which solver and graph really ran, and
  `AMD_SERIALIZE_KERNEL=3` names the kernel behind a GPU trap.
- A fix that has been re-diffed but not run on a card should say so in its
  patch header ("NOT YET RE-VERIFIED ON REAL HARDWARE") until the original
  reproduction has been run against the new binaries.

**Where the numbers come from.** All measurements are from one RX 590 8 GB
(Polaris 30, 36 CUs) on a Ryzen 7 2700X host, Arch Linux with the LTS kernel,
and the ROCm 6.4.4 images built from `rocm6.4.4/`. Other Polaris cards share
the chip design and should behave alike; their clocks, memory and firmware
differ, and none has been tested. Upstream's earlier findings (the rocBLAS
Tensile bugs, MIOpen's grouped-convolution out-of-bounds read, the ROCr
allocator fix and more) are in
[`rocm6.4.4/KERNEL_BUGS.md`](../rocm6.4.4/KERNEL_BUGS.md) and not repeated here.

**Status legend.**

| Status | Meaning |
| --- | --- |
| Fixed | The cause is addressed in source (a patch, or a changed default) and the check passes. |
| Faster | Correct before and after; the change is a performance one. |
| Improved | Partly addressed; the section says what remains. |
| Mitigated | The damage is contained without removing the cause (a cap, a disabled solver, a margin). |
| Worked around | Handled outside the faulty code; the source fix is still to do. |
| Checked | Examined and found correct; nothing was changed. |
| Open | Not solved, or the cause is not understood. |
| all GPUs | Not specific to gfx803: the same source fails or is slow on other architectures too. |

A recurring lesson: on this hardware the common failure is a correct-looking
run with wrong numbers, not a crash. "It runs" was never taken as evidence.

## Index

| Problem | Status | Section | Where |
| --- | --- | --- | --- |
| Attention gradients wrong (shim missed batched GEMMs) | Fixed | 1 | `patches/sgemm-shim-batched/` |
| Shim GEMMs 20x slow at power-of-two sizes | Fixed | 1 | same |
| Shim large GEMMs at 2.5-3 TFLOPS | Faster (3.8-4.8 TFLOPS, pipelined tiles) | 1 | same |
| Binary Winograd convs corrupt memory and hang the GPU | Fixed (solvers disabled) | 2 | `patches/miopen/winograd-bin-gfx803-disable.patch` |
| No fast 3x3 convolution after that | Fixed (new solver) | 2 | `patches/miopen/winograd-f2x3-hip-gfx803.patch` |
| BatchNorm training wrong when H*W is not a multiple of 4 (generic MIOpen bug) | Fixed | 2 | `patches/miopen/bn-spatial-variant1-unaligned-hw.patch` |
| fp16 3x3 convolutions slow (GEMM only) | Fixed (same solver, fp16) | 8 | same |
| rocBLAS fp16 N/T GEMMs 15-25x slow | Fixed (shim fp16 route) | 8 | `patches/sgemm-shim-batched/` |
| rocBLAS fp16 GEMMs in the other layouts 1.3-1.9x slower than the fp32 shim at equal arithmetic | Fixed (fp16 operands on the shim's fast tiles); fp16 SD1.5 UNet step 602 -> 483 ms | 8 | same |
| bf16 reported as supported (no bf16 hardware) | Checked: correct; convs 1.7x slower than fp16 | 8 | none needed |
| PyTorch int8 matmul fails on every call | Fixed (port of upstream's fix) | 5 | `patches/pytorch/` |
| PyTorch layer_norm wrong (warp-size mismatch) | Fixed | 5 | `patches/pytorch/` |
| PyTorch top-k hangs the GPU (warp-size mismatch in its scan) | Fixed | 5 | `patches/pytorch/` |
| PyTorch nonzero, masked indexing, unique, large sorts fail to launch (rocPRIM symbol clash with rocSPARSE) | Fixed | 5 | `patches/pytorch/` |
| PyTorch index_put accumulate drops sums (warp-size mismatch) | Fixed | 5 | `patches/pytorch/` |
| PyTorch embedding backward wrong or faults (warp-size mismatch) | Fixed | 5 | `patches/pytorch/` |
| A GPU-side assert resets the whole GPU instead of raising an error | Open (driver/trap handler) | 5 | none |
| torchvision's roi_align, roi_pool, deform_conv2d and nms had no GPU kernel (the wheel was built CPU-only) | Fixed (torchvision built with its GPU sources) | 5 | `Dockerfile` (`FORCE_CUDA=1`) |
| PyTorch sparse `coalesce()` wrong for dense widths above 160 | Fixed | 5 | `patches/pytorch/gfx803-sparse-coalesce-block-width.patch` |
| PyTorch softmax and log_softmax backward wrong for some large dimensions (shared memory sized for 32 lanes, launched 64 wide) | Fixed | 5 | `patches/pytorch/gfx803-logical-warp-smem.patch` |
| MIGraphX int32 reductions fail to compile for gfx803 | Fixed | 5 | `patches/migraphx/dpp-int32-sum-gfx8.patch` |
| MIGraphX fp16 matrix products accumulate in fp16 (error 0.7-1% instead of 0.02%) | Fixed | 5 | `patches/migraphx/rocblas-compute-fp32-gfx803.patch` |
| ONNX Runtime TopK with sorted = 0 fails above K of about 7,800 | Fixed | 5 | `patches/onnxruntime/topk-safe-small-k-smem-bound.patch` |
| No fused attention in PyTorch on gfx803 | ComfyUI node (inference, fp32 kernels) | 6 | `tools/ComfyUI-gfx803-attention/` |
| Patched HSA runtime page faults | Open; images default to the stock runtime | 4 | `Dockerfile`, start scripts |
| Kernels read stale data from the per-CU vector L1 after a freed address is reused (silent wrong results in rocBLAS, rocSOLVER, torch.linalg) | Fixed (libamdhip64 rebuilt); decode costs about 2.6% | 4 | `patches/clr/` |
| rocSOLVER and rocSPARSE ship no gfx803 code, so their kernels silently never run | Fixed (rebuilt for gfx803) | 4 | `patches/rocsolver/`, `patches/rocsparse/` |
| VRAM exhaustion freezes the desktop | Mitigated (VRAM cap) | 7 | `training/gfx803-train.sh` |
| llama.cpp prompt processing 10x slow | Fixed | 9 | `patches/ollama/llama-cpp/100-*` |
| llama.cpp long-context prompts slow | Fixed | 9 | `patches/ollama/llama-cpp/101-*` |
| llama.cpp q8_0 generation slows with context | Fixed | 9 | `patches/ollama/llama-cpp/102-*` |
| llama.cpp K-quant generation below memory bandwidth | Fixed | 9 | `patches/ollama/llama-cpp/103-*` |
| llama.cpp integer dot products built from byte unpacking | Faster (decode kernels) | 9 | `patches/ollama/llama-cpp/104-*`, `105-*` |
| llama.cpp prompts on emulated int8 (mmq) | Faster (fp32 GEMMs on the shim) | 9 | `patches/ollama/llama-cpp/106-*` |
| llama.cpp DeltaNet concat reads 32 KiB-strided | Fixed (all GPUs) | 9 | `patches/ollama/llama-cpp/107-*` |
| MoE prompts: the GPU idles while each layer's experts cross PCIe | Fixed (all GPUs; needs SDMA copies on gfx803) | 9 | `patches/ollama/llama-cpp-sched/108-*`, `HSA_ENABLE_SDMA=1` in Dockerfile-Ollama |
| MoE generation: one GPU round trip per tensor handed to the CPU experts | Patch kept but not applied: with SDMA on it ends in a GPU reset | 9 | `patches/ollama/unapplied/` |
| MoE generation: the GPU idles during the CPU experts, the shared expert waits for them | Faster (all GPUs) | 9 | `patches/ollama/llama-cpp-sched/112-*` |
| Ollama rejects argument-less tools | Fixed (all GPUs) | 9 | `patches/ollama/ollama/` |
| llama.cpp reserves a vision projector's memory on the wrong device | Fixed (all GPUs) | 9 | `patches/ollama/llama-cpp-server/` |
| Ollama keeps stale free-memory figures on slow-starting GPUs | Fixed (all GPUs) | 9 | `patches/ollama/ollama/` |
| Ollama fills VRAM beside a desktop | Mitigated (margin from amdgpu's counter) | 9 | `start-ollama.sh` |

Paths in the table are relative to `rocm6.4.4/`.

## Glossary

| Term | Meaning here |
| --- | --- |
| Wavefront, warp | The group of GPU lanes that execute together. GCN hardware (gfx803 included) runs 64-lane wavefronts; PyTorch's `C10_WARP_SIZE` is 64 only for gfx9 and 32 for other AMD targets, which is the root of several findings in section 5. |
| LDS | Local data share: the per-workgroup scratch memory on a compute unit (64 KiB on gfx803). |
| Tensile | rocBLAS's kernel generator. Its fp32 GEMM kernels return wrong results on gfx803. |
| Shim | `LD_PRELOAD` library that replaces rocBLAS GEMM entry points with gfx803-correct kernels (`patches/sgemm-shim-batched/`). |
| MIOpen, Find | AMD's convolution library; Find benchmarks solvers for a shape and stores the winner in a find-db. |
| Winograd | A convolution algorithm that trades multiplications for transforms; the binary (hand-written assembly) variants are what hang gfx803. |
| mmq, mmvq | llama.cpp's matmul kernels for prompts (quantized matrix times matrix) and for generation (matrix times a few vectors). |
| dp4a | Integer dot-product instruction; gfx803 has none, so llama.cpp's int8 paths emulate it. |
| KFD, BACO | The kernel's compute driver interface, and the GPU reset the driver falls back to when a queue cannot be preempted. |
| SDMA | The GPU's DMA copy engines, which move data without occupying compute units. |
| HSA runtime | `libhsa-runtime64`, ROCm's user-space runtime beneath HIP; this stack can preload AMD's stock build or a patched one (section 4). |

**Speed, start to finish** (RX 590 8 GB; the front page's table):

- **SD 1.5 image, 20 denoising steps at 512 px**, ComfyUI fp32,
  `--use-split-cross-attention`, classifier-free guidance, timed on a second
  run in the same process: 1758 ms per step on an image from just after the
  first shim speed fix (batched GEMMs covered, the coalesced 16x16 kernel,
  binary Winograd disabled, convolutions on MIOpen's GEMM path), 644 ms now,
  526 ms with the ComfyUI node (section 6): 35 s -> 12.9 s / 10.5 s.
- **SD 1.5 LoRA training step**, 512 px, 4 accumulated images: 12.1 s in the
  first setup with correct results (extended shim, binary Winograd disabled),
  about 3.4 s now (section 1). The original stack ran about 8 s per step with
  wrong attention gradients, GPU hangs and corrupted latents.
- **Ollama / llama.cpp**, Qwen3 8B Q4_K_M, q8_0 KV cache, flash attention,
  `llama-bench` on llama.cpp b11232 as Ollama 0.35.1 fetches it, unmodified
  against patches 100-107: prompt (512 tokens) 14.0 -> 222.8 tokens/s; a
  12,288-token prompt 11.4 -> 172.7 tokens/s (1074 s -> 71 s); generation
  (128 tokens, short context) 27.3 -> 30.5 tokens/s. Qwen2.5-Coder 7B Q4_K_M:
  15.0 -> 230.5 and 29.7 -> 33.2. Generation 8k tokens deep: 20.9 -> 26.5
  tokens/s (Qwen3 8B, before patch 104).

## 1. fp32 GEMM: the SGEMM shim and its gaps

**Background.** rocBLAS's Tensile fp32 GEMM kernels return wrong results on
gfx803 (upstream's finding, see `KERNEL_BUGS.md`). Upstream's fix is an
`LD_PRELOAD` shim that replaces `rocblas_sgemm` and `rocblas_gemm_ex` with a
plain, correct kernel.

**Gap: strided-batched GEMMs.** PyTorch's `bmm`/`baddbmm`, which attention uses,
call `rocblas_sgemm_strided_batched` and `rocblas_gemm_strided_batched_ex`.
The original shim did not intercept them, so they still ran on the broken
Tensile kernels. Attention backward was wrong in 4 of 5 runs, and SD 1.5 LoRA
training loss sat at 3-4 instead of the healthy 0.1-0.2: the model learned from
garbage. The extended shim intercepts both batched entry points. With it, every
op and the whole UNet match a float64 CPU reference, and training loss is
0.12-0.18. Check: `training/gfx803_unet_check.py` (forward and backward of
attention, linear, convolution and normalization, then the full UNet).

**Partition camping.** The shim kernel ran at about 0.7 TFLOPS on most shapes
but collapsed when a leading dimension was a power of two: attention scores at
4096 tokens took 214 ms against 11.8 ms at 4100 tokens. Lanes of a wavefront,
and workgroups dispatched together, strided by the leading dimension, so at
4096 or 8192 floats every access landed in one DRAM channel. The fix makes
threadIdx.x walk the contiguous dimension of each operand, pads the LDS tiles
to 16x17, and orders the grid by row tiles. Results are bit-identical to before
(46 cases: all transposes, plain and batched, beta paths, odd sizes); that
attention GEMM went from 214 to 11 ms.

**Speed.** A profile of training put 88% of GPU time in the shim. Register-tiled
kernels (64x64 and 32x32 output tiles, 4x4 or 2x2 outputs per thread, register
prefetch of the next K slice) handle the larger shapes, chosen per shape by
tile count and K; the 16x16 kernel remains for small outputs. Very wide GEMMs
(more than 65535 column tiles) split across the grid instead of falling back to
rocBLAS. For fp32 with B not transposed, which is most of what PyTorch issues,
faster variants of the two tiles issue the LDS reads for the next K step
before the multiply-adds of the current one, so LDS latency overlaps
arithmetic, and fetch A with float4 or float2 loads. Large GEMMs went from
2.8-3.3 to 3.9-4.8 TFLOPS (2048^3: 3.3 -> 4.8; SD 1.5's linear layers 2.8 ->
3.9-4.3; the 1280-channel convolution at 16x16 as a GEMM 2.5 -> 3.6), and
transposed-B shapes gain 10-25%; LoRA's rank-8 GEMMs and transposed-A shapes
stay within 5% of before. A 20-step SD 1.5 LoRA training run takes 4.0 s per
step against about 4.1 before: the GEMMs are no longer most of a step.
All variants are bit-identical to the reference kernel (46 reference cases and
300 random GEMMs with odd sizes, leading dimensions and offsets).

The transposed layouts got their own tile, `gfx803_sgemm_tile_g`: nn.Linear
reaches rocBLAS with A transposed (40% of an SD 1.5 step's GEMM FLOPs) and its
weight gradient with B transposed. Each operand is staged by its contiguous
index (float4 runs, or single floats where that measured faster), a
transposed A gets a 32-deep K slice, and when A is larger than the 2 MiB L2
the tiles go out in groups of 8 column tiles. Linear layers went from 2.2-3.2
to 3.2-4.5 TFLOPS and weight gradients from 2.5-3.2 to 3.8-5.3. An SD 1.5 step
with the node went from 557 to 526 ms, and a training step from 3.63 to 3.37 s.
Results are bit-identical to the previous kernels on 3000 random GEMMs and on
every one of the 12,000+ GEMM calls of two SD 1.5 runs. One limit the kernel
cannot remove: a leading dimension that is a multiple of 8 KiB (2048 or 4096
floats) puts every row of a K-contiguous operand in one DRAM channel (2048^3
with A transposed: 3.4 TFLOPS against 4.95 untransposed). Such shapes keep the
older kernels, and the fix belongs where the matrix is allocated: padding the
rows by 16 floats takes them to 3.6-4.1. llama.cpp's prompt path does that
(section 9).

fp16 is covered too; see section 8.

## 2. MIOpen's binary Winograd kernels: memory corruption and GPU hangs

**Symptoms.** Two, which turned out to share a cause:

- *Corruption.* Some cached SD 1.5 latents were garbage (standard deviation
  from 80 up to 6.6 million, against a healthy 4.5), and training on them gave
  a loss of 0.6-1.2. Encoding the same images again corrupted a different
  subset; on the CPU, none.
- *Hangs.* Under sustained training load the GPU stalled: 100% busy while power
  fell from about 170 W to 95 W, one dispatch in flight that never completed,
  then a driver timeout and reset that often failed to recover the display.

**Cause.** MIOpen's two binary Winograd solvers, `ConvBinWinogradRxS` and
`ConvBinWinograd3x3U`, are its first choice for nearly every fp32 3x3
convolution on gfx803. Both run hand-written GCN assembly with a persistent
launch (one 512-thread workgroup per CU, all 64 KiB of LDS each, raised wave
priority).

- Every decoded hang (see the tools at the end) had `ConvBinWinogradRxS` as the
  stuck dispatch, with identical code bytes across captures, while the shader
  clock jumped to its top state and power fell: the waves were waiting, not
  executing.
- In the corruption, memory last written by the Winograd kernel *reverts to
  that kernel's data* after later kernels overwrite it. In the VAE encoder, when
  PyTorch's caching allocator reused the encoder output's block for the log-
  variance, every wrong element equaled the old encoder value at the same
  address (333 of 333, 335 of 335, 328 of 328 in three runs); keeping the
  encoder output alive made everything correct. The errors come in runs of 4
  floats, one 16-byte store.
- `MIOPEN_DEBUG_CONV_WINOGRAD=0` removes both: 0 of 10 corrupt encodes, and a
  45-minute full-load stress run that had crashed 5 times out of 5 ran clean.
  Disabling only `ConvBinWinogradRxS` is not enough (7 of 10 still corrupt),
  because MIOpen falls back to `ConvBinWinograd3x3U` from the same family.

What it is not: not a copy race (still wrong after a full synchronize, and wrong
when read by a GPU kernel), not the shim, not the patched runtime, not kernel
ordering (same with `AMD_SERIALIZE_KERNEL=3`). Bounding the kernel's output
buffer descriptor to the real size, and marking its stores `glc`, changed
nothing; every MIOpen gfx8 assembly kernel uses the same descriptor format, so
the persistent launch is what sets these two apart. The kernel-level root cause
is unknown.

**Fix.** [`winograd-bin-gfx803-disable.patch`](../rocm6.4.4/patches/miopen/winograd-bin-gfx803-disable.patch)
makes both solvers not applicable on gfx803, so MIOpen uses its GEMM solvers
(which run on the shim). That alone cost speed: with Winograd off every 3x3
convolution is a GEMM.

[`winograd-f2x3-hip-gfx803.patch`](../rocm6.4.4/patches/miopen/winograd-f2x3-hip-gfx803.patch)
adds `ConvHipWinogradF2x3`: F(2x2,3x3) for 3x3 stride-1 pad-1 fp32
convolutions, forward and backward-data, as three small HIP transform kernels
around one strided-batched GEMM. MIOpen's Find picks it where it is faster
(about 2x on most SD 1.5 UNet convolutions; GEMM stays for the smallest). It is
correct on 17 of 17 shapes in both directions, has lower rounding error than
the GEMM path, and cut training from 4.75 to 4.05 s/step. Weight gradients are
not implemented: LoRA training freezes convolution weights and never needs
them. After installing a build with it, delete MIOpen's user find-db
(`*.ufdb.txt`) once, or old records keep the GEMM choice.

**Check.** Encode a few images with the SD 1.5 VAE on GPU and CPU and compare
`latent_dist.std`, not only the mean: the corruption showed only in the std.

**BatchNorm training at H*W not a multiple of 4.** MIOpen's spatial
BatchNorm (variant 1, NCHW: H*W above 1024 for backward, 4096 or more for
forward training) reads its input four elements at a time across N*H*W.
When H*W is not a multiple of 4 those reads cross from one channel plane
into the next, so the statistics and gradients include elements of the
wrong channel and miss the first elements of each plane. At 33x33 and
65x65 the weight and bias gradients were off by 1.3e-2 to 1.4e-1 (fp32,
against a float64 reference); forward training at 65x65 and 129x129 was
off by up to 4e-3. `MIOpenDriver bnorm ... -V 1` shows the error yet
reports "verification passed", because its tolerance is loose. The bug is
in the kernels' index arithmetic, not gfx803-specific, and is unchanged on
upstream's `release/rocm-rel-7.1` and `release/rocm-rel-7.2`; `develop` has
fixed the backward kernel only.
[`bn-spatial-variant1-unaligned-hw.patch`](../rocm6.4.4/patches/miopen/bn-spatial-variant1-unaligned-hw.patch)
uses scalar reads for those shapes; aligned shapes and NHWC compile to the
same code as before. It also corrects the vector remainder of the backward
kernel for recomputed statistics (`MIOpenDriver -s 0`; PyTorch always passes
saved statistics and never reaches it), which indexed four-element loads at
one-element stride. After it the GPU matches the reference at every tested
shape, at the same speed, with saved and with recomputed statistics.

## 3. Host crashes under load, and what was ruled out

Before the Winograd cause was found, the machine also crashed outright under
the same load: an instant black screen with GPU power still normal and no
kernel message, once ending in a data-fabric sync flood (the next boot logged
"an uncorrected error caused a data fabric sync flood event" with fatal machine
checks in the data fabric). One run on AMD's stock runtime showed both pictures
in one event, the GPU stall and then the host freezing within seconds, so the
host crash is most likely the same failure with the host dying too fast to see
the stall.

Ruled out, each by a run that crashed anyway: the memory overclock (the RAM
had been running at 2733 MT/s against a JEDEC 2133; still crashed at 2133),
PCIe Gen3 (crashed on Gen2), the shim's batched kernel, the patched runtime,
and a 150 W power cap (delayed it, did not prevent it). Steady graphics load at
full power (glmark2, 45 minutes) was clean, so sustained power alone is not the
trigger. Capping the shader clock at 1365 MHz (`pp_dpm_sclk` states 0-4)
prevented it in both the stress test and a full training segment, but only
made the failure rarer; disabling the binary Winograd kernels is what removed
it, at full clocks and power.

## 4. The HSA runtime: patched or stock

The base image carries upstream's patched `libhsa-runtime64` (a "va-reuse-
defer" change to how freed GPU virtual addresses are reused; see
`KERNEL_BUGS.md`). On this card it page-faults ("Page not present", writes
from the texture cache) in workloads that free and reallocate GPU memory in
particular patterns: a synthetic GEGLU op test, and wide GEMMs followed by a
4.9M-element transposed copy, every run, where AMD's stock runtime passes. A
storm of such faults in several processes overflowed the interrupt ring and
froze the desktop until a reboot.

Ollama's llama-server crashed on the same faults mid-prompt (a 17k-token Qwen
Code request on Qwen3.5 9B: ten `VM fault ... write from 'TC*'` lines, then an
interrupt ring overflow). ComfyUI has faulted twice on the second generation
after a model reload, which fits the same pattern but is not proven.

The images therefore preload AMD's stock runtime, which the ROCm base image
still carries as `libhsa-runtime64.so.1.15.60404` beside the patched
`.so.1.15.0`. Ollama, SDXL and Qwen-Image generation, fp16 and training are all
validated on it (one intermittent training fault is listed in section 10). `GFX803_RUNTIME=patched` (start scripts)
and `gfx803-train --runtime patched` load the patched one for comparison. The
cost is the bug the patch was written for: MIOpen's CK reduction kernels can
read stale data when freed addresses are reused at once (ONNX Runtime's
`ReduceSum` test), a path PyTorch, sd-scripts and llama.cpp do not take. What
in the va-reuse-defer change faults on gfx803 is not investigated yet.

### A kernel can read stale data from the vector L1 across dispatches

Symptom: a rocBLAS call returns `rocblas_status_success` with wrong numbers.
`rocblas_sgemm_strided_batched` with m=n=k=8 and batch 4 comes out with a
relative error of 0.94, bit-identical on every run, at the same point of a
sequence of about 500 small GEMM, GEMV and TRSM calls that each
`hipMalloc`, copy in, compute, copy out and `hipFree`. The same sequence with
`hipFree` turned into a no-op passes every case. A single set of buffers that is
refilled with `hipMemcpy` before each call, with no `hipFree` at all, fails as
well for small operands (see the measurements below). `torch.linalg` (LU, solve,
inverse, QR, SVD, eigh) goes through the same calls and returned wrong factors
that varied from run to run, and MIOpen's reduce kernels returned wrong sums in
a sequence of calls on recycled buffers.

Cause: the command processor of gfx803 (MEC firmware 730) does not
invalidate the per-CU vector L1 cache at a dispatch boundary. ROCm expresses
cross-dispatch coherence only through the fence scopes in the AQL packet
header, and the LLVM memory model for GFX6 to GFX9 relies on the packet
processor to act on them, so no code ever issues the cache invalidate itself.
A kernel on a CU whose L1 still holds lines of a page that something else has
rewritten since then reads the old bytes. The rewrite is a copy: copies of 16 KiB
or less are shader copies that write through L2, and an address comes back
either because the application refills the same buffer or because ROCr's
fragment heap hands a freed 4 KiB page to the next `hipMalloc` of the same size,
with no ioctl, unmap or TLB flush in between. Operands much larger than the 16 KiB
L1 push the old lines out on their own, which is why big GEMMs were never
affected. The wrong values are what
was last at that address (a zero-padded and a NaN-padded buffer give different
garbage at the same case, so this is not an out-of-bounds read), serialising
launches does not cure it, and neither do SDMA, the SGEMM shim or the workspace
size.

Fix: `patches/clr/` rebuilds `libamdhip64` from ROCm/clr at the rocm-6.4.4 tag
with one PM4 `ACQUIRE_MEM` packet, published in its own AQL ring slot ahead of
every kernel dispatch, that invalidates the vector L1 and the scalar cache
(`CP_COHER_CNTL` = `TCL1_ACTION_ENA | SH_KCACHE_ACTION_ENA`). Only the one
library is replaced; the Ollama image carries the identical stock file, so
the same rebuilt file fits it. `CLR_GFX8_TC_INVALIDATE=0` switches the packet off in the same
binary, which is how the fix is measured against itself.

Measured on the RX 590 with the stock HSA runtime, the same binary and only
`CLR_GFX8_TC_INVALIDATE` toggled:

| Test | Packet off | Packet on |
| --- | --- | --- |
| rocBLAS probe, 511 free-and-reallocate cases | stops at case 225 | passes |
| `torch.linalg` through the rebuilt rocSOLVER, 204 cases up to n=257 | wrong | 204 of 204 |
| MIOpen reduce kernels, 52 shapes (`rocm6.4.4/tools/reduce-harness`) | 12 fail | 0 fail |
| same, with every call on one recycled address | 12 fail | 0 fail |
| stock rocBLAS sgemm, fixed buffers, 40x72x800, 500 iterations | 499 iterations corrupt (about 6,000 wrong elements each) | 0 |
| same, 32x32x32 | 499 iterations corrupt | 0 |
| llama.cpp `test-backend-ops -o SOLVE_TRI` (triangular solve), 24 cases | 6 fail (matrices over 64x64) | 0 |
| same, 16x16x16 | 492 iterations corrupt | 0 |
| same, 768x768x3072 | 0 | 0 |

The regression set (shim bit identity, conv checks, UNet check, rocPRIM and top-k
ops, 20 training steps) passes without a GPU fault, and six 300-step LoRA training
runs and the LLM generation canaries (dense and MoE, temperature 0) give the same
results with the packet on and off. Decode on a 7B Q4_K model drops by 2.3% to 2.7% (33.5 to
about 32.7 tokens/s) and prompt processing is unchanged. Which caches the packet
touches matters: the vector L1 alone, the L1 plus scalar cache, and the L2 write-back
plus invalidate pair all cure the probe at about the same decode cost; an L2
invalidate without the write-back does not, with or without the L1 bit, and an
empty word does not, and the full amdgpu set (adding the instruction cache)
costs 7.3% of decode.

The cost is per dispatch, not per byte: back-to-back empty kernels on one
stream take 4.3 us each with the packet and 2.65 us without it, the same in a
replayed HIP graph, and a 4096-block kernel pays the same 1.5 us. Workloads
made of many small kernels pay the most: MoE decode issues about 1,450
dispatches per token, an fp16 SD 1.5 step about 1,430 in ~400 ms.

Open: why the L1-plus-L2-invalidate word fails while its neighbours pass, and
why fp64 never fails. The probes in `tools/tc-staleness/` that go through
PyTorch (`probe.py`, `soak.py`, the copy-engine and graph-replay probes) are
clean on this stack with the packet off as well, so only the rocBLAS probe
isolates the bug, together with the rocBLAS and reduce programs in the table.
That MIOpen's reduce failures and rocBLAS's wrong results on small fp32 GEMMs
(`KERNEL_BUGS.md`) are this bug follows from the table; whether the patches written
for them (the va-reuse-defer runtime change, the small-shape rocBLAS patch) are
still needed has not been tested. The faults of the patched HSA runtime and the
intermittent training fault in section 10 have not been explained: six training
runs with the fix were clean, which at the old fault rate of 2 in 16 is not
evidence either way.

### rocSOLVER and rocSPARSE had no gfx803 code

ROCm's prebuilt `librocsolver` and `librocsparse` contain device code for gfx9
and newer only. A kernel launch with no code object for the card is dropped
without an error, and rocSOLVER does not check the launch status, so
hipSOLVER returned success. The info and pivot arrays then kept whatever the
allocator handed out (an `info` of 1094295552 is the bit pattern of 12.0f), and
PyTorch's pivot unpacking indexed with the garbage until the GPU faulted.
`torch.sparse.mm` returned a result with relative error 1.0.

Both libraries are now built from source for gfx803 against the image's
rocBLAS in their own Dockerfile stages and replace the shipped files
(`patches/rocsolver/`, `patches/rocsparse/`). `torch.sparse.mm` has a
relative error of 1.4e-7 and the `torch.linalg` check above passes. Cholesky
and `lstsq` need MAGMA, which this PyTorch is built without, and are not
supported. Sparse `coalesce` is still wrong for widths over 160.

## 5. PyTorch fixes and workarounds in the base image

**int8 matmul.** `torch._int_mm` (int8 x int8 -> int32, used by anything that
quantizes linear layers to int8, such as ComfyUI's int8 backends) failed on
every call with `HIPBLAS_STATUS_INVALID_VALUE`: on ROCm, PyTorch sends it to
hipBLASLt unconditionally, and hipBLASLt has no gfx803 kernels.
`TORCH_BLAS_PREFER_HIPBLASLT=0` does not help, because this one path never
checks it. [`rocm6.4.4/patches/pytorch/`](../rocm6.4.4/patches/pytorch/) routes
it to rocBLAS whenever hipBLASLt is not the preferred backend, as PyTorch's
floating-point GEMMs already do. The fix comes from upstream's ROCm 10 line
(Schaka/rocm-gfx803 commit 5061404), adapted to torch 2.8. Check: 18 of 18
shapes and transpose layouts bit-exact against an int64 CPU product; 4 TOPS at
2048^3.

**layer_norm.** PyTorch's HIP layer_norm was wrong on gfx803: called
directly in fp32 (`torch.ops.aten.native_layer_norm`), forward and backward
disagreed with a float64 CPU reference in 80 of 80 runs across SD 1.5's
shapes, with forward errors up to 8.5 on unit-scale outputs, and every
transformer block uses it. The cause is a warp-size mismatch. Device code
takes its warp size from `C10_WARP_SIZE` in `c10/macros/Macros.h`, which is 64
only for `__GFX9__` and 32 for every other AMD target, gfx803 included,
although gfx803 runs 64-lane wavefronts. layer_norm's host code sized launches
and shared memory with `at::cuda::warp_size()`, the hardware's 64: the
vectorized forward kernel ran 64-wide rows of threads in which only lanes 0-31
reached the cross-warp reduction, so the mean and variance left out half of
each row, and the backward kernel allocated shared memory for 4 warps where
its block reduction writes 8.

[`gfx803-layer-norm-warp-size.patch`](../rocm6.4.4/patches/pytorch/gfx803-layer-norm-warp-size.patch)
makes layer_norm's launchers use the warp size the device code was compiled
with (32 on a 64-lane GPU that is not gfx9). Every kernel in the file is
correct with 32-wide blocks, because its shuffles stay within 32 lanes. After
it, native layer_norm matches the CPU in 80 of 80 runs (forward error 2.5e-6,
backward 2.7e-7), and so do all three gradients over odd sizes and both
backward kernels. Defining `C10_WARP_SIZE` as 64 for gfx8 instead also fixes
layer_norm, but it changes every kernel compiled against the macro and some
depend on 32: group_norm's weight and bias gradients then come out about 100%
wrong whenever the spatial size is 1 and there are more than 128 rows
(`GammaBeta1dBackwardCUDAKernel2`, a fixed 32x16 block reduced with
full-wavefront shuffles).

**top-k.** `torch.topk` hung the GPU on gfx803, and the driver then reset it
(`SQ instruction intr detected ... trap true`, then `qcm fence wait loop
timeout`, `Failed to evict process queues`, a BACO reset and `VRAM is lost`).
Any sampler that filters tokens with top-k or top-p on the GPU hits it on its
first token; Chatterbox TTS did. With `AMD_SERIALIZE_KERNEL=3` the HIP
runtime names the kernel (`at::native::sbtopk::gatherTopK`) and reports
`HSA_STATUS_ERROR_EXCEPTION` before the reset. gatherTopK finds each kept
element's output slot with a binary prefix scan
(`aten/src/ATen/cuda/ScanUtils.cuh`) that votes with a 64-bit ballot over the
whole wavefront and has lane 0 store its count, but numbers warps by
`C10_WARP_SIZE`. On gfx803 lanes 32-63 form a warp whose count nobody
writes, thread 0 sums uninitialized shared memory into the offsets, and the
kernel's `CUDA_KERNEL_ASSERT(writeIndex < outputSliceSize)` executes a trap.
A trapped wave on gfx803 never finishes, so the queue cannot be preempted and
the driver resets the GPU instead of the process receiving an error. The scan
as written gets 100 of 100 random blocks wrong on the card, and 0 with the
hardware's warp width
([`tools/correctness-suite/pytorch/`](../tools/correctness-suite/pytorch/)
holds this reproduction and the checks quoted below).
[`gfx803-topk-scan-warp-size.patch`](../rocm6.4.4/patches/pytorch/gfx803-topk-scan-warp-size.patch)
numbers the scan's warps by the hardware wavefront on ROCm. After it, top-k
matches the CPU in 798 of 798 cases (sizes up to 1M and 152k vocabularies,
fp32, fp16 and int64, largest and smallest, sorted and unsorted, heavy ties),
and Hugging Face's top-k, top-p and min-p filters keep the same tokens as on
the CPU in 150 of 150 trials.

**rocPRIM kernels that would not launch.** `nonzero`, boolean-mask indexing
(`x[mask]`, `masked_select`), `unique`, `randperm`, the backward of
`nn.Embedding`, `cumsum` on integers, `searchsorted` and sorts of more than
4096 elements failed on gfx803 with `HIP error: invalid device function`,
at any input size. rocPRIM is header-only, so each library built with it
carries its own copy of every rocPRIM kernel it uses and exports the
kernel's host-side handle. ROCm's rocSPARSE, which libtorch_hip links,
exports 31 of the same handles as libtorch_hip. The dynamic linker binds both
libraries to one address, each library registers that address with its own
code object, and HIP keeps the first registration, rocSPARSE's. AMD's
rocSPARSE has no gfx803 code object, so the launch finds no kernel. On GPUs
rocSPARSE supports, its identical copy runs and nothing is visibly wrong.
The same kernels work in a standalone gfx803 program, and rocPRIM kernels
that rocSPARSE does not export work inside torch.
[`gfx803-rocprim-private-namespace.patch`](../rocm6.4.4/patches/pytorch/gfx803-rocprim-private-namespace.patch)
compiles torch's HIP code with `-Drocprim=at_hip_rocprim`, the ROCm
counterpart of the `CUB_WRAPPED_NAMESPACE` that PyTorch uses against the same
clash with CUDA's CUB; libtorch_hip then exports no rocPRIM symbol and shares
no kernel name with rocSPARSE. A check of these ops against the CPU went from
67 to 118 of 118 cases (with the index_put fix below).

**index_put with accumulate.** `index_put_(..., accumulate=True)`, which
`x[idx] += v`-style scatters and the backward of advanced indexing use,
returned wrong sums whenever indices repeated: off by up to 4.9 with 1000
indices into 1000 slots and by 175 with 3 million, against a CPU error below
3e-4. It is layer_norm's mismatch again: the launcher in `Indexing.cu` sizes
blocks and grids by `at::cuda::warp_size()` (64) while its kernels index in
`C10_WARP_SIZE` (32) warps, so half of each block's sorted indices are never
summed and lanes 32-63 repeat lanes 0-31.
[`gfx803-index-put-warp-size.patch`](../rocm6.4.4/patches/pytorch/gfx803-index-put-warp-size.patch)
launches both index_put kernels (plain and quantized) with the compiled warp
size, as the layer_norm patch does. After it the GPU matches the CPU; at 3
million indices it is closer to a float64 reference than the CPU is (4e-5
against 2.5e-4).

**embedding backward.** The backward of `F.embedding` and `nn.Embedding` with
at most 3072 indices (the path every text encoder and token-embedding LoRA
takes) returned wrong weight gradients, off by 2.7 to 26 in fp32 against a
float64 reference, and some shapes (31 indices, dim 64) read out of bounds and
ended the process with a GPU memory access fault. The launcher runs
`embedding_backward_feature_kernel` with 64-wide rows and sizes its shared
memory for them, but the kernel lays its rows out `C10_WARP_SIZE` (32) apart,
so each row overwrites half of the next and the last one overwrites the
indices it reads its destination rows from. The kernel elects a leader among
rows with a ballot whose lane bits stand for row numbers, so each row must be
one hardware wavefront and the index_put-style fix (a 32-wide launch) would
break it;
[`gfx803-embedding-backward-row-width.patch`](../rocm6.4.4/patches/pytorch/gfx803-embedding-backward-row-width.patch)
lays the rows out by the launched width, `blockDim.x`, instead. After it the
GPU matches the CPU (largest difference 5.1e-6, as on an NVIDIA T4).

These are four instances of one pattern: code that checks for gfx9 when it
means "64-lane GPU" leaves gfx6-8 with a 32-lane `C10_WARP_SIZE` on 64-lane
hardware, and a kernel goes wrong wherever it meets a value sized for the
hardware. Fourteen of the 21 files under `aten/` that use the macro also call
`at::cuda::warp_size()`; the audit is target PT-2 in
[`OPTIMIZATION_TARGETS.md`](OPTIMIZATION_TARGETS.md).

torchvision is built with its GPU sources (`FORCE_CUDA=1`; its setup compiles them only when it
can see a GPU, which it cannot during `docker build`). Without that, `roi_align`, `roi_pool`,
`ps_roi_align`, `deform_conv2d` and `nms` have no GPU implementation and fail on GPU tensors.
`tools/correctness-suite/pytorch/torchvision_ops_check.py` compares them with the CPU, forward and
backward; `nms` and `batched_nms` also match from 1 to 20,000 boxes. torchaudio is still built
without its GPU ops.

Four more kernel bugs, each reproduced on the card before the fix and clean after it:

- **Sparse `coalesce()`.** The kernel strides over features with the 32-lane `C10_WARP_SIZE` while it is launched with a
  64-wide block, so features 160 and up of every 256-wide tile are never written. `gfx803-sparse-coalesce-block-width.patch`
  strides by `blockDim.x`. `tools/correctness-suite/pytorch/sparse_coalesce_check.py`: 23 failing cases before, none after.
- **Softmax and log_softmax backward.** The block reductions and the persistent softmax launch with the hardware warp width
  but index their shared memory by 32, so some large dimensions read uninitialised shared memory
  (`log_softmax` in fp16 at dimension 9672: backward error 0.11, row sums off by 0.12).
  `gfx803-logical-warp-smem.patch` sizes those launches by the logical width. `softmax_smem_check.py`: 13 failing
  cases before, none after. The speed of softmax before and after has not been measured.
- **MIGraphX.** The DPP reduction for int32 emits `v_add_u32 ... row_shr`, which does not assemble for gfx8, so every
  int32 `reduce_sum` fails to compile (`dpp-int32-sum-gfx8.patch`). Its GEMM asks rocBLAS for fp16 compute on gfx803,
  so fp16 `dot` accumulates in fp16 and is off by 0.7% to 1% instead of 0.02%
  (`rocblas-compute-fp32-gfx803.patch`). `migraphx_check.py` runs both through the gpu target.
- **ONNX Runtime TopK.** With `sorted = 0` the kernel asks for shared memory that grows with K; above about 7,800 it
  exceeds the 64 KiB of LDS and the ROCm provider raises `hipErrorInvalidValue`. `topk-safe-small-k-smem-bound.patch`
  sends those requests to the radix kernel (`ort_topk_smem_bound_check.py`: 20 of 20 cases pass).

The workarounds below are loaded into every Python process through `sitecustomize` in the base
image; the modules are in [`rocm6.4.4/runtime/python/`](../rocm6.4.4/runtime/python/).
They are workarounds with their source fixes still to do.

- **accelerate on a build without torch.distributed.** accelerate 1.15 imports
  `torch.distributed.tensor` unconditionally, which crashes on this
  single-GPU build; the image stubs the check (a model on such a build cannot
  hold distributed tensors).

## 6. ComfyUI node with gfx803 kernels for attention, GroupNorm and GEGLU

PyTorch has no fused attention on gfx803, so every SD 1.5 attention call
stores its full score matrix (8 x 4096 x 4096 floats at the 64x64 level).
[`tools/ComfyUI-gfx803-attention/`](../tools/ComfyUI-gfx803-attention/) is a
ComfyUI custom node that replaces three operations of ComfyUI's SD 1.5 path
with HIP kernels from [`tools/`](../tools/):

- attention for head sizes 40, 64 and 80, in one pass with an online softmax
  (`gfx803_fused_attention.hip`: a block of four waves shares each 32-key K/V
  tile across 128 query rows), hooked into `F.scaled_dot_product_attention`
  and ComfyUI's split and optimized attention (fp16 inputs are computed in
  fp32); head size 160 and calls that need gradients stay on PyTorch;
- GroupNorm with float64 accumulation (`gfx803_group_norm_silu.hip`), fp32
  and up to 128 channels per group;
- GEGLU, the split, exact-erf GELU and product in one pass
  (`gfx803_geglu.hip`).

All three match a CPU float64 reference (attention within 3.5e-6 relative,
GroupNorm and GEGLU within 1e-5, over odd shapes, 2-D/3-D and non-contiguous
inputs). A full ComfyUI SD 1.5 fp32 txt2img, 512 px and 20 steps, sent 400
attention, 320 GEGLU and 1220 GroupNorm calls through the node, and its image
differs from a CPU run's by at most 1/255 (mean 0.001/255). The same
workflow at fp16 takes 8.5 s per warm image with the node and 10.5 s
without it. With the node, a 512 px fp16
denoising step (ComfyUI's default precision on this card) takes 342 ms
instead of 392, and its attention 79 ms instead of 129. The kernel reads the
fp16 projections in ComfyUI's [b, tokens, heads*D] layout and writes its
output there, with no transposed or fp32 copies around the call. Per call, with
classifier-free guidance (batch 2, 8 heads): self-attention at 64x64 12.5 ms
against 50 ms for ComfyUI's split attention and 58 ms for PyTorch's SDPA,
which also allocates 2.3 GB for it (the fused kernel 10 MB); SDXL's
self-attention at 4096 tokens (head size 64, 10 heads) 10.6 ms per batch
item; GEGLU 1.6-1.7x faster; GroupNorm 1.0-1.35x faster. At head size 160 the
fused kernel is slower than split attention (0.79 against 0.34 ms), so that
level is left to ComfyUI.

The attention kernel's multiply-adds are written as `a * b + c`, never
`fmaf()`. gfx803 issues the fused `v_fma_f32` at a quarter of the rate of
`v_mac_f32` and `v_mad_f32` (measured: 894 against 3,540 G operations per
second over the card), and the compiler turns `a * b + c` into `v_mac_f32`
because fp32 denormals are flushed on gfx8. With `fmaf()` in its two inner
products the same kernel runs 2-3x slower. The unfused form rounds each
product, which shows only on ill-conditioned inputs: with logits scaled 50x
the kernel lands 2e-4 to 1.1e-3 from a float64 reference, within 2.5x of
what a CPU fp32 computation of the same inputs gets (2e-4 to 4.5e-4); at
realistic magnitudes every head size stays within 3e-6.
`tools/correctness-suite/pytorch/attention_check.py` checks both.

The node is not built into the images. `tools/ComfyUI-gfx803-attention/install.sh`
compiles the three libraries (with the base image's `hipcc` when the host has
none) and copies the node into `rocm6.4.4/ComfyUI/custom_nodes/`.

## 7. VRAM on an 8 GB card that also drives a display

Uncapped, PyTorch keeps freed memory and grows until the card is full: an SD
1.5 run that needs about 6.1 GB took 7.99 of 8 GB. The driver then moves
buffers to system memory and back (steps slowed from 4.3 to 60 seconds), and
the compositor fails to place its framebuffers ("Failed to pin framebuffer with
error -12"), which can freeze a monitor until a reboot. `gfx803-train` caps
PyTorch at what the card's other users hold at startup plus 1 GB for the HIP
runtime, measured from amdgpu's own VRAM counter (ROCm's count misses the
desktop's framebuffers). Even capped at 90% training caused brief pin failures
on a busy desktop, and 512 px SD 1.5 training does not fit beside a typical
desktop at all, so training needs the display manager stopped.

## 8. fp16, and what it makes possible (SDXL LoRA training)

**Correctness.** fp16 computes correctly on this stack: 63 operation cases
(every GEMM layout, batched, SD1.5's convolutions forward and backward data,
norms, softmax, attention) match a float64 CPU result from the same fp16
inputs to 2-9e-4, and rocBLAS accumulates fp16 GEMMs in fp32. The SD1.5 and
SDXL UNets in fp16 match fp32 to about 1e-3.

**Three slow paths, all fixed.**

- fp16 3x3 convolutions had only MIOpen's GEMM path, 1.2-1.75x slower than
  fp32 Winograd forward and far slower backward. `ConvHipWinogradF2x3` now
  takes fp16 too (fp16 workspace, fp32 transform math): fp16 convolutions run
  at fp32's speed, and the SD1.5 UNet forward in fp16 takes 394 ms against
  452 ms before (fp32: 390 ms).
- rocBLAS's fp16 GEMMs with op(A) = N, op(B) = T run 15-25x slower than the
  other layouts on gfx803 (20 ms against 1 ms for 1280x4096x320; 510 ms against
  23 ms for a 16-batch attention GEMM). That layout is the backward pass of
  1x1 convolutions, MIOpen's other GEMM-based backward convolutions and
  attention, and fp16 SD1.5 training ran at 7.7 s/step because of it. The SGEMM
  shim now takes those calls (and all strided-batched fp16 calls) onto its own
  kernels with fp16 storage; fp32 results are unchanged, bit for bit.
- rocBLAS's fp16 GEMMs in the other layouts (TN, NN and TT: `nn.Linear` and its
  relatives) run a Tensile kernel that widens each fp16 operand to fp32 inside
  the inner loop, about 1.6 times the vector instructions per multiply-add of the
  shim's fp32 tiles, so they ran 1.3-1.9x slower than fp32 GEMMs of the same
  shape on the shim. The shim's fast tiles now read fp16 operands directly (a
  wide load, one widening convert, fp32 accumulation, one rounding at the
  store) and take all four layouts and the batched calls. Across 31 GEMM
  cases they are 1.0-2.4x faster than before (median 1.5x) with the same error
  against float64 (4.2e-4 at worst); an fp16 product that overflows gives
  infinities in the same places as rocBLAS. The fp16 SD1.5 UNet step (512 px,
  CFG batch of 2, the attention node loaded) takes 483 ms against 602 ms: the
  convolutions go from 228 to 147 ms (their Winograd GEMMs are fp16 batched
  GEMMs) and the linear layers from 176 to 137 ms; attention does not run on
  these tiles (section 6). A seeded UNet step differs from a fp32 run by 0.6-0.9%
  before and after. Two of the shim's dispatch rules were tuned on fp32
  operands and lost at fp16, so they now apply to fp32 only: single-float
  staging of B for op(A) = T with K >= 2048 (SD1.5's feed-forward output
  projections, 1280x512x5120: 2.1 -> 3.5 TFLOPS) and the 64x64 tile for
  M = 32 (the Winograd GEMMs at the smallest latent: 0.86 -> 0.63 ms). The
  outputs are bit-identical; the fp16 step goes from 415 to 399 ms. `tools/unet_breakdown.py` prints the split by operation.
- An fp16 operand widened right after its global load makes the compiler wait
  for that load inside the fetch, one load at a time, so each K slice of the
  fp16 tiles paid four or five memory round trips that nothing overlapped.
  The shim keeps the fetched values fp16 until the next slice's LDS store, and
  a one-wavefront 64x64 tile (an 8x8 block per thread, no barriers) takes the
  fp16 strided-batched NN calls with batch >= 8 and M >= 512. Outputs are
  bit-identical on 478 cases. fp16 GEMM time over the SD call logs drops 12%
  (SDXL 2426 -> 2120 ms); the fp16 UNet step goes from 392 to 360 ms at
  SD 1.5 and from 3.34 to 2.98 s at SDXL. The same tile on single GEMMs runs
  up to 3.3x slower (nothing in its workgroup hides its loads), and at fp32 it
  is erratic, so it takes only those batched fp16 calls.
  A two-wavefront 64x128 tile takes the fp16 TN calls with at least three
  rounds of its tiles over the CUs: both TN operands are contiguous in K, the
  one-wavefront tile's 8-deep pieces read a quarter of each 64-byte line, and
  the two wavefronts share 16-deep slices instead (SDXL TN 10240x2048x1280
  10.4 -> 8.6 ms, 5120x8192x640 13.9 -> 10.1 ms). Bit-identical; fp16 UNet
  step SDXL 2.98 -> 2.84 s, SD 1.5 359 -> 355 ms.
- The node's fused attention runs one wave per SIMD, so every instruction
  besides its multiply-adds costs it time, and fetching the softmax
  probabilities from other lanes with `ds_bpermute` (128 per 32-key tile)
  cost 10% of the kernel. The lanes now hold the score tile so that each
  row's probabilities lie in one 16-lane DPP row, and the P*V multiply-adds
  read them through a DPP row rotation (`v_mac_f32_dpp`) at no instruction
  of their own. Attention time per step: SDXL 504 -> 416 ms (step 2.84 ->
  2.75 s), SD 1.5 91 -> 79 ms (step 355 -> 342 ms). The log-sum-exp the
  training backward uses is bit-identical; outputs differ by at most one
  fp16 rounding step, at the same distance from a float64 reference.

**Training.** SD1.5 LoRA, 512 px, batch 1 with gradient accumulation 4,
steps 5-20 of a 20-step run with a warm find-db, card peak VRAM:

| | gradient checkpointing | without |
| --- | --- | --- |
| fp32 | 2.87 s/step, 5.8 GB | 2.07 s/step, 7.1 GB |
| fp16 mixed precision (`gfx803-train`'s default) | 2.47 s/step, 3.6 GB | 1.73 s/step, 4.9 GB |

gfx803 runs fp16 arithmetic at the fp32 rate, so the 14-16% comes from the
halved memory traffic of the fp16 GEMM tiles and the widening deferred to the
LDS store; the 2.2 GB saved is enough to train without gradient
checkpointing beside a desktop. The 20-step losses of the four runs
(0.16-0.22) are within run-to-run variation; no longer run compared fp16
against fp32 training quality. `gfx803-train` trains in fp16 mixed precision
(the frozen model in fp16, fp32 LoRA weights, loss scaling) unless given
`--precision fp32`.

**Attention in training.** PyTorch's attention keeps the full score and
probability matrices for the backward pass: one SD1.5 self-attention at the
64x64 level (8 heads, 4096 tokens) holds 2.6 GB extra, and an fp32 UNet
forward and backward at 512 px runs out of 8 GB on that alone, which is why
512 px training needs gradient checkpointing (an extra forward per step).
The node's attention library now has a backward that recomputes the
probabilities from Q, K and the forward's per-row log-sum-exp, so its memory
grows with the token count. One kernel computes S and dP once per tile pair
and accumulates dK and dV in registers; gfx803 has no global float atomic add,
so each key block writes its dQ partial sums to a scratch slab (336 MB for the
64x64 level at batch 1, from PyTorch's allocator) and a second kernel adds the
slabs in a fixed order. At 4096 tokens, head size 40, it takes 18.3 ms against
59 ms for PyTorch's forward and backward with 2.6 GB of matrices. Its
gradients match a float64 reference within 1e-5 (fp32) and 5e-4 (fp16
inputs). `gfx803-train` runs sd-scripts with `--sdpa`, which routes every
attention with head size 40, 64 or 80 through it
(`training/gfx803_attn_launch.py`; `GFX803_ATTN_TRAIN=0` turns it off). SD1.5
LoRA, fp32, 512 px, batch 1, 4 accumulated, 40 steps, seed 42: 3.37 s/step at a
7.6 GB card peak before, 2.87 s/step at 5.6 GB with it, and 2.03 s/step at
6.9 GB with `--no-checkpointing`, which the fused attention makes possible;
the loss matches the previous run at every step.

**SDXL.** With fp16, SDXL LoRA training fits in 8 GB on a headless machine.
Measured training steps (LoRA dim 8, gradient checkpointing, cached text
encoder outputs unless noted, ~7 GB PyTorch cap):

| setup | per image | card VRAM |
| --- | --- | --- |
| 768 px, fp16 UNet | 5.9 s | 6.9 GB |
| 768 px, fp8 UNet (`--fp8_base`) | 5.9 s | 4.6 GB |
| 1024 px, fp8 UNet | 10.7 s | 6.4 GB |
| 768 px, fp8, text encoders trained too | 7.0 s | 6.0 GB |

1024 px with an fp16 UNet, and 1024 px with the text encoders trained, do not
fit. fp8 weight storage changes the UNet output by about 6% against fp32;
that is fp8's rounding, identical on any GPU (the GPU matches a CPU run with
the same fp8-rounded weights to 1e-3). A real 12-step `sdxl_train_network.py`
run at 768 px in fp16 completed with loss 0.20 and a 7.9 GB card peak.

**bf16.** gfx803 has no bf16 hardware, but PyTorch reports compute capability
8.0 and `torch.cuda.is_bf16_supported()` returns True, so ComfyUI, sd-scripts
and other tools may choose bf16 on their own. On this stack that is safe:
casts, elementwise math, reductions, softmax, matmuls (plain, batched,
linear), 3x3 convolution, group and layer norm and attention all match the CPU
to bf16's own rounding (about 2e-3), with no faults or hangs. Upstream's
ROCm 10 line reported bf16 hanging the card under vLLM (Schaka/rocm-gfx803,
`vllm/NOTES.md`); that does not reproduce here. bf16 matmuls run at fp16/fp32
speed, but 3x3 convolutions take about 1.7x as long (2.54 ms against 1.43 ms
fp16 for 320 channels at 64x64), because the Winograd solver above covers fp32
and fp16 only. Prefer fp16 for half precision on this card.

## 9. llama.cpp and Ollama

Ollama 0.35.1 builds llama.cpp b11232; its HIP backend compiles for gfx803
with no source changes, and computes correctly (llama.cpp's own operation tests
pass against its CPU backend). It was slow for gfx803-specific reasons
that share a theme: kernels tuned for GPUs with more instructions spill
registers, or run out of instruction throughput, on gfx803. The first five
bullets are the original findings (patches 100-105); the rest build on them.

- **Prompt processing at 15 tok/s** (Qwen3 8B Q4_K_M). The quantized matrix
  multiply used the GCN tile table, sized for GPUs with `v_dot4`. gfx803 has
  no integer dot instructions, the emulated dot products need more registers,
  and the wide K-quant tiles spilled hundreds of VGPRs (565 for Q4_K at 64
  columns). A gfx803 table drops those tiles: 157 tok/s. Rejected on
  measurement: hand-written dp4a emulation (the compiler's version already uses
  fewer instructions), routing through the fp32 shim (1.84 TFLOPS, below the
  fixed path's 2.41), and rocBLAS's fp16 GEMM (+10%, but fp16 accumulation and
  unvalidated kernels).
- **Long-context prompts 4x slower than short ones.** The flash-attention tile
  kernel uses packed half2 arithmetic on every AMD GPU. gfx803 has fp16 but no
  packed half2 instructions, so the path gains nothing and its extra registers
  made the wide tiles spill (648 VGPRs for the 16x4 tile). On the kernel's fp32
  path they spill 43. Qwen3 8B, 8k tokens deep: 34 -> 127 tok/s.
- **Generation with an 8-bit KV cache slowed with context** (27.9 tok/s empty,
  20.9 at 8k). The vec kernel that generates with a quantized cache needs 229
  VGPRs on gfx803, so one wave runs per SIMD and its memory latency goes
  unhidden; time followed the number of query heads, not the bytes read. It
  could not be narrowed (spills) or extended to share K/V loads across the
  query heads of a grouped-query model (spills, slower). A new gfx803 kernel
  serves all query heads of a K/V head from one read: one lane per K row for
  the scores, one lane per output dimension for the weighted sum, 66-104 VGPRs.
  Generation 8k deep: Qwen3 8B 20.9 -> 24.4 tok/s, Qwen2.5-Coder 7B 24.8 ->
  27.2. With it, the 8-bit cache is as fast as f16 or faster from about 4k
  tokens, at half the VRAM, which on 8 GB decides whether a 16k context fits.
- **Generation below the memory bandwidth for K-quant models.** Generating a
  token reads every weight once, so it should run at the card's bandwidth, and
  for q8_0 and f16 weights it does (222-229 GB/s for a single-token matmul).
  The K-quants that make up a Q4_K_M model reached 127-165 GB/s, with few
  registers in use (34 VGPRs): without integer dot instructions their dot
  products are long ALU sequences, and the kernel is instruction-bound. One
  warp over two rows per block instead of two warps per row halves the per-row
  reduction and launch overhead: q4_K +11-27%, q3_K +31%, q6_K +9%. q5_K, q8_0,
  iq4_xs and q5_1 measured slower that way and keep the old shape. Generation:
  Qwen3 8B 27.9 -> 31.4 tok/s empty and 24.2 -> 26.5 at 8k, Qwen2.5-Coder 7B
  30.3 -> 34.6 and 27.0 -> 30.1.
- **Integer dot products assembled from unpacked bytes.** Every quantized
  dot product goes through `ggml_cuda_dp4a`, which newer GPUs run as one
  instruction. gfx803 has none, so the compiler unpacks four bytes of each
  operand and multiplies them one at a time; in the decode kernels keeping
  those bytes live pushes the 2-8 token variants (speculative or parallel
  decoding) to 120-256 VGPRs, and the q8_0 flash-attention kernels spill.
  gfx803's SDWA operand selects let `v_mul_i32_i24` multiply byte i of one
  register by byte i of another directly. `104-gfx803-sdwa-dp4a.patch` uses
  that in the decode kernels: those above 128 VGPRs go from 56 to 1, spills
  from 2 to 0, and 4-8 token decode matmuls get up to 1.6x faster. Single
  tokens are unchanged and 2 tokens lose 1-11%. In the prompt kernels (mmq)
  and in the dot products whose operand comes from a 16-entry table (IQ4_NL,
  IQ4_XS, MXFP4, NVFP4) SDWA measured slower (q8_0 prompt tiles 21%, IQ4_XS
  decode up to 2.6x), so those keep the compiler's code; why SDWA loses there
  is not known. In Ollama at a 5.2k-token
  context, Qwen3 8B and Qwen2.5-Coder 7B (Q4_K_M) generate 6-7% faster
  (24.9 -> 26.6 and 28.4 -> 30.5 tok/s) with prompt speed unchanged, and
  Qwen3.5 9B IQ4_XS is unchanged (151 tok/s prompt, 29 tok/s generation);
  output is bit-identical in all three. Where the Q4_K_M generation gain comes
  from is not measured, since their single-token matmuls are unchanged; the
  q8_0 flash-attention kernels losing their spills is the likely candidate.
  `105-gfx803-q5k-decode.patch` loads q5_K's scales with one 16-byte load
  instead of four small ones (q5_K decode matmul 233 -> 224 us, measured with
  104).
- **Prompts on emulated int8.** Even with the tile table, mmq runs q4_K at
  about 2.4 effective TOPS, a third of the card's fp32 rate, and rounds the
  activations to 8 bits. `106-gfx803-cublas-prompt.patch` sends quantized
  matmuls with 128 or more columns to cuBLAS in fp32, which the base image's
  SGEMM shim runs. Upstream picks fp16 compute with fp16 accumulation for
  quantized weights on every AMD GPU, which on gfx803 goes to rocBLAS; the
  patch uses fp32 there. Weight rows of 4096 or 14336 floats (multiples of
  8 KiB) put a whole GEMM tile in one DRAM channel, so the weights are
  dequantized straight into rows padded by 16 floats (14336 x 512 x 4096:
  2.8 -> 4.1 TFLOPS). Weights whose fp32 copy would exceed 512 MiB keep mmq:
  an output head over a 152k vocabulary (reached only when every token's
  logits are computed) needs 2.5 GB, and that run ended in a GPU memory fault
  on the stock runtime while the SGEMM alone at that shape ran clean. Results:
  llama-bench pp512 Qwen3 8B 163.5 -> 222.8, Qwen2.5-Coder 7B 166.9 -> 230.5;
  Ollama with a ~5.2k-token prompt 149 -> 215 and 107 -> 136 tok/s;
  generation unchanged. KL divergence against the CPU backend drops (Qwen2.5-
  Coder 0.00152 -> 0.00094, Qwen3 0.0103 -> 0.0086). MoE expert matmuls
  (MUL_MAT_ID) keep mmq: each expert sees too few tokens per batch.
- **DeltaNet's convolution input read 32 KiB apart.** Qwen3.5/3.6 and
  Qwen3-Next build it as concat(conv state, transpose(qkv)), and llama.cpp's
  generic concat kernel read the transposed operand with every lane one row
  (32 KiB) from the next: 27 ms per call, 0.83 s of a 2048-token batch of
  Qwen3.6 35B-A3B. `107-concat-transposed-src1.patch` moves it through 32x32
  shared-memory tiles (1.14 ms). This is not specific to gfx803. With 106 and
  107, Qwen3.6 35B-A3B (experts in system RAM) in Ollama reads a ~5.5k-token
  prompt at 236 tok/s against 189 before, and generates at 23.8 tok/s against
  22.2.
- **Decode re-quantized the same input for every product.** Each quantized
  matrix-vector product converts its fp32 input to q8_1 in its own kernel, and
  most layers start with several products of one input. A dispatch costs at
  least 4.3 us on this card however small the kernel, so the repeats add up
  over the ~1,450 dispatches of a MoE token. `111-mmvq-reuse-quantized-src1.patch`
  keeps the converted input until a node writes over it: generation +1-2%
  (Qwen3 8B 30.1 -> 30.3-30.8 tok/s), identical outputs. Not specific to
  gfx803.
- **MoE prompts waited for every layer's expert copy.** With most experts
  in system RAM, each MoE layer of a prompt batch runs on the GPU after the
  scheduler copies that layer's experts there, and the GPU sat idle for the
  whole PCIe transfer: removing the copies (timing only) took Qwen3.6
  35B-A3B from 295 to 376 tok/s at 2048-token batches and from 167 to 388
  at 512. `108-moe-expert-prefetch.patch` copies the next MoE layer's
  experts into a second device slot on a second stream while the current
  layer computes, for batches large enough that nearly every expert is
  used. On gfx803 that overlap needs `HSA_ENABLE_SDMA=1`: ROCm's runtime
  otherwise copies host to device with blit kernels, which serialize with
  compute (a 512 MiB copy next to a GEMM loop: no overlap at all, against
  37 ms of 41 hidden with SDMA). Dockerfile-Ollama sets it. llama-bench
  pp2048 with all experts in RAM: 193 / 250 / 295 -> 243 / 321 / 367 tok/s at
  512 / 1024 / 2048-token batches. In Ollama with Chatterbox loaded beside
  it (3.2 GiB of the model on the card), prompts of 2k and 8k tokens went
  124 -> 144 and 122 -> 135 tok/s. Generation is unchanged. Logits are
  unchanged too (llama-perplexity with and without: PPL 12.1753, mean KLD
  0), and 767 GiB of byte-compared SDMA round trips next to running GEMMs
  found no mismatch. The scheduler is compiled into `libggml-base`, which
  `Dockerfile-Ollama` installs from its CPU build, so the patch is applied
  before that build (`patches/ollama/llama-cpp-sched/`) and the build checks
  the installed library for it. In the image, through Ollama's default
  512-token batches, a 3,100-token prompt reads 165 -> 196 tok/s and a
  11,900-token prompt 174 -> 193 tok/s, with no GPU fault. On a short prompt
  the greedy text with prefetch on follows the text with it off for 295
  characters and then diverges at a near-tie; with prefetch off the text is
  identical to the unpatched scheduler's.
- **Each GPU tensor a CPU expert split reads cost a full round trip.** In
  generation with experts in RAM, every layer hands its hidden state, router
  weights and expert ids to the CPU, and the scheduler copied each with a
  synchronous copy: about 25 us per 4 KB copy-and-wait on an RX 590, about
  130 per token. `109-sched-batched-host-copies.patch` queues those copies
  and waits once per split: Qwen3.6 35B-A3B tg128 23.8 -> 24.4 tok/s, logits
  unchanged. It is not applied: with the scheduler patched in `libggml-base`
  and `HSA_ENABLE_SDMA=1`, generation with the experts in host memory ends in
  `ring sdma0 timeout` and a GPU reset (two reproductions, with expert prefetch
  on and off), and it gives no speedup in the image (28 to 31 tokens/s with and
  without it). The cause of the SDMA timeout is not understood. The rest of a
  generated token's time is in docs/OPTIMIZATION_TARGETS.md (LLM-8).
- **The GPU idled through every CPU expert split, and the shared expert
  waited for it.** In the same generation, each layer runs attention and
  the router on the GPU, then the routed experts on the CPU (about 0.44 ms),
  and the scheduler synchronizes the GPU stream before the CPU starts. The
  Qwen3.5/3.6 and Qwen3-Next graphs built the shared expert after the routed
  experts, so it ran after them, about 80 us per layer, although its input
  is the router's. Leaving it out of the graph (timing only) took Qwen3.6
  35B-A3B from 23.8 to 25.2 tok/s. `112-moe-shared-expert-overlap.patch`
  adds the gated shared expert to the graph right after the router weights,
  and the scheduler computes such a GPU split in two parts: an event after
  the last node the CPU split reads (with views resolved to the tensor they
  point into, and the consuming nodes right after it kept in the same part
  so the fused top-k routing kernel stays fused) lets the CPU copy its
  inputs and start while the GPU runs the shared expert. The next GPU split
  waits for an event recorded after that part rather than synchronizing the
  stream, which on HIP costs about 15 us more right after a graph launch.
  llama-bench with every expert in RAM, tg128: 24.5 -> 25.2 tok/s with
  `HSA_ENABLE_SDMA=0`, 24.6 -> 25.6 with SDMA; through Ollama 25.3 -> 26.2.
  Logits are unchanged: llama-perplexity at 1- and 16-token batches gives
  the same PPL to the last digit with the overlap on and off, mean KLD 0,
  same top token 100%, with SDMA on and off, and Ollama's greedy output is
  identical. Unlike 109, it keeps the scheduler's synchronous copies and
  changes only what they wait for (an event instead of the whole stream); no
  SDMA fault occurred in any run. `GGML_SCHED_EARLY_HOST_INPUTS=0` disables
  it.

Upstream llama.cpp has no equivalent of the third fix; issue
[#27796](https://github.com/ggml-org/llama.cpp/issues/27796) reports the same
quantized-cache slowdown on RDNA4 and was closed as expected behavior.

Three more bugs are not specific to gfx803 but matter most on a small card
that also drives the display:

- **Ollama re-encodes a tool with no arguments** as `"properties": null`, which
  llama.cpp's server rejects with HTTP 400. Coding agents that declare such
  tools, Qwen Code among them, fail on their first request.
- **llama.cpp's server misplaces a vision projector's memory reservation.**
  When fitting a model it adds the projector's worst-case memory under the
  backend registry's device index rather than the model's. Ollama loads the
  CPU backend first, so on a one-GPU machine the reservation is inverted: a
  projector on the GPU gets none (the model fills the card and image encoding
  then needs memory that is not free), and a projector on the CPU holds back
  about 1.1 GB of VRAM for nothing. With Qwen3.5 9B on the RX 590, a 1080p
  image took the card to 8074 of 8192 MiB; the compositor could no longer pin
  its framebuffers and applications lost their GPU contexts.
- **Ollama keeps stale free-memory figures on GPUs that start slowly.** Its
  refresh before each load starts a small llama-server with a 3-second
  deadline and keeps the old values on a miss. The ROCm library takes about
  3 s to start on gfx803 and a refresh about 8 s, so Ollama kept counting the
  memory of unloaded models and, for example, put a vision projector on the
  CPU because it seemed not to fit on an empty card. The deadline is now twice
  the first discovery's duration (3-30 s).

ROCm's own free-memory figure also leaves out the display's framebuffers on
this card, so llama.cpp's 1 GiB fitting margin is mostly taken by the desktop
before a model loads. `start-ollama.sh` sets the margin from amdgpu's in-use
VRAM counter plus 512 MiB of headroom.

Model choice matters as much as kernels on 8 GB. A hybrid model such as
Qwen3.5 9B (one full-attention layer in four) runs a 64k context fully on the
GPU beside a desktop; and some uploads keep the output tensor in bf16, which
for a 248k-token vocabulary is 2 GB of a 6.4 GB model, read for every token.
Converting only that tensor to q8_0
([`rocm6.4.4/ollama-tools/requant-output-head.py`](../rocm6.4.4/ollama-tools/requant-output-head.py))
freed 0.95 GB and made generation 11% faster, with the same top token as the
original 98% of the time (KL divergence 0.0014).

A mixture-of-experts model goes further on the same card. Qwen3.6 35B-A3B
(Q4_K_M, 22 GB) activates about 3B parameters per token: llama.cpp keeps its
attention, shared weights and a q8_0 KV cache for the full 262k context on
the GPU and most routed experts in system RAM. Before patches 106-108, on a
5.5k-token prompt it read 189 tok/s and generated 22 tok/s with DDR4-2800 (158
and 19 at 2133); the bullets above give the later figures.
What decided those numbers:

- **The bundled MTP draft head costs speed.** Checking a draft runs the
  experts of every drafted token, and those are read from system RAM, so
  generation fell from 19.3 to 18.4 tok/s with it on; its f16 draft cache
  (512 MiB at this context) also took VRAM from the experts.
- **Batch size trades prompt speed for expert VRAM.** Prompt processing
  copies the RAM-resident experts to the card once per batch, so 1024-token
  batches read prompts 31% faster than 512 at a 1% decode cost; at 2048 the
  compute buffer pushes experts off the card and decode drops to 12 tok/s.
- **Anonymous memory beats the page cache for the experts.** Loading without
  mmap made prompt reading 10% faster with generation unchanged.
- **Threads: one per core.** 16 threads on the 8-core CPU dropped decode to
  2.7 tok/s.

Generation moves only about 12 GB/s from RAM, well under the memory's
bandwidth, so the limit is the CPU's expert math and the roughly 80 GPU/CPU
handoffs per token. Against the CPU backend the model's logits differ by a
KL divergence of 0.014 (same top token 93.9%), in line with dense models
that are known to be right on this stack (Qwen3 8B 0.010, Qwen3.5 9B 0.007).
It is the first local model here that completes Qwen Code's tool-calling
tasks.

Each patch header in `rocm6.4.4/patches/ollama/` has the full measurements and
a reproduction.

## 10. Open questions

- Why the binary Winograd kernels fail on gfx803 (they are disabled, not
  fixed). Candidates: their persistent launch, or wrong cache/release fields in
  their converted code-object metadata on gfx8.
- Why the patched HSA runtime's va-reuse-defer change page-faults on gfx803
  (GEGLU op test, wide GEMMs then a transposed copy, llama-server mid-prompt),
  and whether it causes ComfyUI's faults after a model reload. The images run
  AMD's stock runtime until this is understood; the reproducers are
  `gfx803_unet_check.py --only GEGLU` and the wide-GEMM-then-transposed-copy
  sequence described in section 4, on
  `GFX803_RUNTIME=patched` with the desktop stopped (a fault storm can freeze it).
- Whether PyTorch's group_norm computes wrong weight and bias gradients on
  gfx9 for a spatial size of 1 and more than 128 rows:
  `GammaBeta1dBackwardCUDAKernel2` reduces a 32-wide tile with shuffles over
  `C10_WARP_SIZE` lanes, which upstream sets to 64 there (section 5). That
  would be an upstream bug; not tested, as there is no gfx9 card here.
- Why llama.cpp's SDWA dot product (patch 104) is slower than the compiler's
  code in the prompt kernels, for table-lookup operands and for 2-token decode.
- Whether the decode kernels (mmvq) beat the prompt kernels (mmq) for 9-16
  tokens on gfx803; llama.cpp switches at 8.
- fp16 over full-length training runs and in ComfyUI (only short runs so far).
- An intermittent GPU memory fault in SD 1.5 LoRA training on AMD's stock HSA
  runtime. Two of sixteen runs of 120-300 steps (same image and data) died, at
  steps 32 and 74, with `Memory access fault ... Page not present` and amdgpu
  `VM fault ... read from 'TC'` lines, without a GPU reset; the loss was healthy up
  to that point and nothing was NaN. To reproduce, run a 300-step
  `gfx803-train` job on the SD 1.5 dataset (about 3.5 s/step, 7.2-7.6 GB peak
  VRAM) and check `journalctl -k` for `VM fault` afterwards; expect to need
  several runs.
  Ruled out or not supported: the card's power mode (one fault in manual mode, one
  in auto mode, with the PCIe link at 2.5 GT/s during training, which is the
  driver's choice under low bus traffic); VRAM pressure (three 300-step runs under
  a 6800 MiB PyTorch cap, which forces the allocator to free cached segments and
  retry one to three times per run, all completed with no fault). Runs with
  `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1`, with full `AMD_LOG_LEVEL=3`
  logging (both about 6.3 s/step) and with kernel-name-only logging
  (`AMD_LOG_LEVEL=3 AMD_LOG_MASK=0x80`, 3.9 s/step) also all completed, but they
  change timing, so they do not show where the fault comes from.
  What the faults look like: the faulting reads come from several compute units at
  once, at addresses whose low 36 bits look like one device pointer (in one fault,
  consecutive pages) while bits 36-39 differ from lane to lane; in the
  first fault two of the addresses were exactly 8 GiB minus one page apart.
  That points at a kernel that builds a pointer from a wrongly read or stale
  upper half, or at a mapping that is briefly absent, and not at a plain
  out-of-bounds index. The kernel is not identified. Sections 2 and 4 describe
  the known sources of faults of this kind on this card; the stale vector L1 of
  section 4 would produce a stale upper half, and this fault has not been
  re-run with that fix.
- Why images are slow to read: a 1080p image (about 2000 tokens) takes 56 s
  with Qwen3.5 9B, against about 15 s for 2000 text tokens.

## Tools

- [`tools/gpu-hang-capture/`](../tools/gpu-hang-capture/): a 2-second GPU
  monitor, a watcher that captures a stuck compute queue's registers and
  in-flight dispatch packets within seconds (before the driver's reset erases
  them), and a decoder that names the stuck kernel. This is how the Winograd
  kernels were identified. Never read `/sys/kernel/debug/kfd/mqds` while queues
  are active: on this kernel it oopses and turns a GPU hang into a host lockup.
- [`tools/ComfyUI-gfx803-attention/`](../tools/ComfyUI-gfx803-attention/): the
  optional ComfyUI node with gfx803 attention, GroupNorm and GEGLU kernels
  (section 6); its kernel sources are in `tools/`.
- [`tools/tc-staleness/`](../tools/tc-staleness/): probes for the stale vector L1
  of section 4; `probe_rocblas.py` is the one that fails without the fix.
- [`tools/correctness-suite/`](../tools/correctness-suite/): the operation
  checks, including the BatchNorm and activation backward sweeps and
  `pytorch/linalg_check.py`.
- [`rocm6.4.4/training/gfx803_unet_check.py`](../rocm6.4.4/training/gfx803_unet_check.py):
  the op-by-op and whole-UNet check against a float64 CPU reference.
