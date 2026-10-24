# Findings: what breaks on gfx803, why, and what this project does about it

AMD removed gfx803 (Polaris: RX 470, 480, 570, 580, 590) from ROCm after 6.0.
The code still compiles for it, but much of the stack no longer produces
correct results, or runs far below what the hardware can do. This document
collects what this project found while making SD 1.5 and SDXL LoRA training,
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
- Paths are relative to the repository root.
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
and the ROCm 6.4.4 images built from this repository. Other Polaris cards share
the chip design and should behave alike; their clocks, memory and firmware
differ, and none has been tested. Schaka's earlier findings (the rocBLAS
Tensile bugs, MIOpen's grouped-convolution out-of-bounds read, the ROCr
allocator patch, which this project does not build (section 4), and more) are in
[`docs/KERNEL_BUGS.md`](KERNEL_BUGS.md) and not repeated here.

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
| PyTorch int8 matmul fails on every call | Fixed (port of Schaka's ROCm 10 fix) | 5 | `patches/pytorch/` |
| PyTorch layer_norm wrong (warp-size mismatch) | Fixed | 5 | `patches/pytorch/` |
| PyTorch top-k hangs the GPU (warp-size mismatch in its scan) | Fixed | 5 | `patches/pytorch/` |
| PyTorch nonzero, masked indexing, unique, large sorts fail to launch (rocPRIM symbol clash with rocSPARSE) | Fixed | 5 | `patches/pytorch/` |
| PyTorch index_put accumulate drops sums (warp-size mismatch) | Fixed | 5 | `patches/pytorch/` |
| PyTorch embedding backward wrong or faults (warp-size mismatch) | Fixed | 5 | `patches/pytorch/` |
| A GPU-side assert resets the whole GPU instead of raising an error | Fixed (host setting: compute-wave save and restore on, the driver default) | 3 | host: `amdgpu.cwsr_enable` |
| torchvision's roi_align, roi_pool, deform_conv2d and nms had no GPU kernel (the wheel was built CPU-only) | Fixed (torchvision built with its GPU sources) | 5 | `Dockerfile` (`FORCE_CUDA=1`) |
| PyTorch sparse `coalesce()` wrong for dense widths above 160 | Fixed | 5 | `patches/pytorch/gfx803-sparse-coalesce-block-width.patch` |
| PyTorch softmax and log_softmax backward wrong for some large dimensions (shared memory sized for 32 lanes, launched 64 wide) | Fixed | 5 | `patches/pytorch/gfx803-logical-warp-smem.patch` |
| MIGraphX int32 reductions fail to compile for gfx803 | Fixed | 5 | `patches/migraphx/dpp-int32-sum-gfx8.patch` |
| MIGraphX fp16 matrix products accumulate in fp16 (error 0.7-1% instead of 0.02%) | Fixed | 5 | `patches/migraphx/rocblas-compute-fp32-gfx803.patch` |
| ONNX Runtime TopK with sorted = 0 fails above K of about 7,800 | Fixed | 5 | `patches/onnxruntime/topk-safe-small-k-smem-bound.patch` |
| No fused attention in PyTorch on gfx803 | ComfyUI node (inference; its GroupNorm and GEGLU are fp32 only) | 6 | `tools/ComfyUI-gfx803-attention/` |
| Schaka's patched HSA runtime (va-reuse-defer) page-faults on gfx803 | Not built: the images run AMD's stock runtime, and the bug it targeted is the stale vector L1 below | 4 | git history up to `0e67b7d` |
| Kernels read stale data from the per-CU vector L1 and scalar cache after a freed address is reused (silent wrong results in rocBLAS, rocSOLVER, torch.linalg, MIOpen reductions) | Fixed (libamdhip64 rebuilt); decode costs about 2.6% | 4 | `patches/clr/` |
| Device printf and assert() hang (hostcall's doorbell signal never changes) | Fixed (ockl.bc rebuilt) | 4 | `patches/device-libs/` |
| No-return atomics on fine-grained host memory store their operand instead (user kernels) | Open (hardware; use the returned value) | 4 | none |
| Non-inlined device functions read implicit kernel arguments from address 0: PyTorch's mixed fp16/fp32 binary ops crash, assert() faults at its trap | Fixed (clang and libamd_comgr rebuilt, PyTorch rebuilt with them) | 4 | `patches/llvm/` |
| Flat (generic-pointer) accesses to LDS are dropped when M0 is not set (gfx8) | Fixed (same compiler) | 4 | `patches/llvm/` |
| rocSOLVER and rocSPARSE ship no gfx803 code, so their kernels silently never run | Fixed (rebuilt for gfx803) | 4 | `patches/rocsolver/`, `patches/rocsparse/` |
| VRAM exhaustion freezes the desktop | Mitigated (VRAM cap) | 7 | `training/gfx803-train.sh` |
| llama.cpp prompt processing 10x slow | Fixed | 9 | `patches/ollama/llama-cpp/100-*` |
| llama.cpp long-context prompts slow | Fixed | 9 | `patches/ollama/llama-cpp/101-*` |
| llama.cpp q8_0 generation slows with context | Fixed | 9 | `patches/ollama/llama-cpp/102-*` |
| llama.cpp K-quant generation below memory bandwidth | Fixed | 9 | `patches/ollama/llama-cpp/103-*` |
| llama.cpp integer dot products built from byte unpacking | Faster (decode kernels) | 9 | `patches/ollama/llama-cpp/104-*`, `105-*` |
| llama.cpp prompts on emulated int8 (mmq) | Faster (fp32 GEMMs on the shim) | 9 | `patches/ollama/llama-cpp/106-*` |
| llama.cpp DeltaNet concat reads 32 KiB-strided | Fixed (all GPUs) | 9 | `patches/ollama/llama-cpp/107-*` |
| MoE prompts: the GPU idles while each layer's experts cross PCIe | Fixed (all GPUs; needs SDMA copies on gfx803) | 9 | `patches/ollama/llama-cpp-sched/108-*`, `HSA_ENABLE_SDMA=1` and `ROC_CPU_WAIT_FOR_SIGNAL=1` in Dockerfile-Ollama |
| MoE generation: one GPU round trip per tensor handed to the CPU experts | Patch kept but not applied: no speedup in the image; with SDMA on it reset the GPU on kernel 6.18.55 | 9 | `patches/ollama/unapplied/` |
| GPU resets: `ring sdma0 timeout` with SDMA copies on | Fixed in the image (trigger removed): a kernel waiting for scratch, the page-table update that maps it, and a user SDMA queue polling for a kernel behind it deadlock engine 0 once a second process tips page-table jobs onto that engine; CLR now waits for dependencies on the CPU (prompt speed unchanged). The engine's behaviour remains | 3 | `ROC_CPU_WAIT_FOR_SIGNAL=1` in Dockerfile-Ollama |
| MoE generation: the GPU idles during the CPU experts, the shared expert waits for them | Faster (all GPUs) | 9 | `patches/ollama/llama-cpp-sched/112-*` |
| MoE generation: ~1,425 GPU dispatches per token at ~4.3 us each | Faster (all GPUs) | 9 | `patches/ollama/llama-cpp/113-*` to `116-*` |
| MoE prompts: the DeltaNet recurrence spends a wave per state column (26% of a Qwen3.6 prompt batch) | Faster (all GPUs; prompts +26%) | 9 | `patches/ollama/llama-cpp/117-*` |
| Ollama rejects argument-less tools | Fixed (all GPUs) | 9 | `patches/ollama/ollama/` |
| llama.cpp reserves a vision projector's memory on the wrong device | Fixed (all GPUs) | 9 | `patches/ollama/llama-cpp-server/` |
| Ollama keeps stale free-memory figures on slow-starting GPUs | Fixed (all GPUs) | 9 | `patches/ollama/ollama/` |
| Ollama fills VRAM beside a desktop | Mitigated (margin from amdgpu's counter) | 9 | `start-ollama.sh` |

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
| HSA runtime | `libhsa-runtime64`, ROCm's user-space runtime beneath HIP; the images run AMD's stock build (section 4). |

**Speed, start to finish** (RX 590 8 GB; the front page's table):

- **SD 1.5 image, 20 denoising steps at 512 px**, ComfyUI fp32,
  `--use-split-cross-attention`, classifier-free guidance, timed on a second
  run in the same process: 1758 ms per step on an image from just after the
  first shim speed fix (batched GEMMs covered, the coalesced 16x16 kernel,
  binary Winograd disabled, convolutions on MIOpen's GEMM path), 644 ms now,
  526 ms with the ComfyUI node (section 6): 35 s -> 12.9 s / 10.5 s.
- **SD 1.5 LoRA training step**, 512 px, 4 accumulated images: 12.1 s in the
  first setup with correct results (extended shim, binary Winograd disabled),
  about 2.5 s now in fp16 mixed precision, `gfx803-train`'s default, and
  2.9 s in fp32 (section 8). The original stack ran about 8 s per step with
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
gfx803 (Schaka's finding, see `docs/KERNEL_BUGS.md`). Schaka's fix is an
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

**Fix.** [`winograd-bin-gfx803-disable.patch`](../patches/miopen/winograd-bin-gfx803-disable.patch)
makes both solvers not applicable on gfx803, so MIOpen uses its GEMM solvers
(which run on the shim). That alone cost speed: with Winograd off every 3x3
convolution is a GEMM.

[`winograd-f2x3-hip-gfx803.patch`](../patches/miopen/winograd-f2x3-hip-gfx803.patch)
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
[`bn-spatial-variant1-unaligned-hw.patch`](../patches/miopen/bn-spatial-variant1-unaligned-hw.patch)
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

### GPU resets from the SDMA engine (`ring sdma0 timeout`)

With `HSA_ENABLE_SDMA=1`, which runs ROCm's copies on user SDMA queues,
loading a model in Ollama can deadlock SDMA engine 0. Ten seconds later the
kernel logs `amdgpu: ring sdma0 timeout, signaled seq=N, emitted seq=N+2`
(sometimes N+1 or N+3) and
resets the card through BACO. VRAM is lost, every ROCm process on the card
loses its work, and the desktop session usually goes with it. The `Failed to
suspend process` and `cp is busy, skip halt cp` lines that follow come from
the reset itself: the driver's queue-suspend path cannot take the reset lock
that the reset already holds, so they say nothing about the state before the
stall.

The deadlock has three parts, all read from the registers, rings and queues
of the stalled GPU and from a kernel trace of the seconds before it:

1. A kernel on one of the runner's compute queues needs more scratch memory
   than the queue has. The CP stops the queue in front of that dispatch and
   asks ROCr for more; ROCr's event thread allocates and maps the memory and
   then restarts the queue (`DynamicQueueEventsHandler`,
   `AcquireQueueMainScratch`). On an RX 590 the first such request is
   36 CUs x 32 waves x 2 KiB = 2,359,296 bytes of VRAM.
2. Mapping that memory is a page-table update, and amdgpu runs page-table
   updates on an SDMA engine's own ring. Each process's page-table jobs go to
   whichever of the two engines the drm scheduler scores lower, and the score
   counts the job queues of every process that has used that engine.
3. Engine 0 is busy with the runner's user SDMA queue, which sits on a
   dependency wait: when HIP issues a copy that must wait for a kernel that is
   still running, CLR hands the kernel's completion signal to ROCr, and ROCr
   writes a POLL_REGMEM packet that polls until the signal reaches zero and
   never gives up. The kernel it waits for is queued behind the dispatch from
   step 1. The engine stays in the poll even after the queue's time slice has
   expired (`SDMA0_RLC0_CONTEXT_STATUS` shows the expired bit) and never runs
   its own ring, so the page-table update never runs, the scratch never
   arrives, and the poll never ends.

A second process decides where step 2 lands. With Ollama alone on the card,
engine 0 also carries the driver's buffer-move queues, so the runner's
page-table jobs go to engine 1, the scratch is mapped, and nothing stalls
(a traced run sent all 42 of the scratch thread's jobs to engine 1). Any other
process with a GPU address space, even one that only called `hsa_init`,
leaves its page-table queues on engine 1 and raises that engine's score. The
runner's jobs then lean to engine 0, and the scratch mapping lands behind the
poll. That is why the size and activity of the other process do not matter: a
queue-less, memory-less process triggers it as reliably as a text-to-speech
server holding 3.8 GB.

The reset therefore reproduces when all of these hold:

- Ollama loads a model entirely into VRAM, and the model has a kernel that
  needs scratch (Gemma 4 E2B does);
- `HSA_ENABLE_SDMA=1`;
- another ROCm process exists on the card;
- the KFD hardware scheduler runs the queues (`amdgpu.sched_policy=0`, the
  default). Without it the queue-less neighbour did not trigger a stall in
  120 loads; a neighbour holding 3.8 GB still caused two in about 200, one of
  which ended in an SDMA write fault (`VM protection fault`, client SDM0, no
  PASID) and the other in a reset when the stall returned after the driver
  restored the queues.

Without the fix below, all 21 such runs reset the card, 17 of them on the
first load, with compute-wave save and restore on and off
(`amdgpu.cwsr_enable`). Ollama alone ran 80 loads without a reset, and loads
that leave part of the model on the CPU ran 260. CLR passes a dependency to
the GPU only while it is still pending at submit, so the hang depends on
timing: with `AMD_LOG_LEVEL=4` the host is slow enough that no wait reaches
the SDMA queue, and a logged run does not reproduce it. ROCm itself leaves
user SDMA queues off on gfx8 ("User SDMA queues are unstable on gfx8", ROCr
`GpuAgent::InitDma`); `HSA_ENABLE_SDMA=1` overrides that default.

Ruled out, each by a run that reset anyway: the memory clock (pinned at its
top level, automatic, and forced to its lowest), the amount of VRAM in use and
where buffers land, eviction (GTT use stays flat up to the hang), the failed
pins of read-only mmap'd model pages (`use_mmap: false` resets too), a NOP
before ROCr's own POLL_REGMEM and FENCE packets (a rebuilt ROCr resets the
same way), a larger CLR signal pool (`ROC_SIGNAL_POOL_SIZE=512`), and this
repository's CLR cache-invalidate packets (`CLR_GFX8_TC_INVALIDATE=0`).
Polaris SDMA also has a separate, documented hang on POLL_REGMEM and FENCE
packets in the driver's ring, made rarer by a NOP before each such packet
(mainline c343a27ed908, tagged for stable); the resets above occur with that
workaround in the running kernel. Kernel 6.18.57 also stops putting two
POLL_REGMEM packets on sdma0 for every KFD VRAM clear (stable 30dacb56f072,
upstream 87ceb8cba73d), which on 6.18.55 reset the card during the MoE runs
in section 9 (about one heavy prompt run in ten to fifteen; generation with
patch 109, two of two); on 6.18.57 both run clean. The devcoredump does not
show the stall: its ring contents and pointers are read after the reset.

The fix: Dockerfile-Ollama sets `ROC_CPU_WAIT_FOR_SIGNAL=1` next to
`HSA_ENABLE_SDMA=1`. With it CLR waits for every pending dependency on the CPU
and submits the copy without one, so no poll ever holds an SDMA engine, and
the scratch mapping in step 2 runs whichever engine it lands on. The setting
is CLR's own; it is CLR's default when HIP does not dispatch directly. With it
the reproduction ran 720 loads without a reset:

- beside the queue-less process;
- beside a process churning 256 MiB of VRAM every 50 ms;
- beside the text-to-speech server;
- with MoE loads;
- with the image as built, beside the text-to-speech server and the
  queue-less process together, and with compute-wave save and restore on.

Greedy output is identical with and without it on four models
(Qwen3.6-35B-A3B, Qwen3 8B, Gemma 4 E2B, Gemma 3n E2B). Qwen3.6-35B-A3B with
its experts in host memory (2.8k-token prompt, medians of two interleaved
sets of five requests) keeps its prompt speed, 338 tokens/s, and generates
24.3 tokens/s against 24.5, inside the run-to-run spread.

The alternatives cost more or cover less:

- `HSA_ENABLE_SDMA=0` drops that prompt speed to 263 tokens/s (-22%).
- Waiting on the CPU only for copies' dependencies also avoids the reset, but
  drops it to the same 262. Why CLR's switch keeps the overlap and the
  narrower wait does not is not known.
- Moving copies up to 4 MiB onto blit kernels (`GPU_FORCE_BLIT_COPY_SIZE=4096`)
  avoids this reproduction only because every copy of these loads is smaller
  than 4 MiB. Larger copies keep their GPU-side waits.
- Running Ollama alone on the card avoids it only by the scheduler's choice of
  engine, not by design.

The engine's refusal to leave a polling queue is in the SDMA hardware or
firmware and cannot be fixed here; the image keeps dependency waits out of
the SDMA queues, which removes the third part of the cycle.

To reproduce, without the setting: start an idle ROCm process (for example a
program that calls `hsa_init` and sleeps). Then send chat requests to the
Ollama image that alternate between Gemma 4 E2B and Gemma 3n E2B (both fit
in VRAM) with `keep_alive: 0`, so that every request loads a model. The reset
comes on a Gemma 4 E2B load, usually the first. With
`ROC_CPU_WAIT_FOR_SIGNAL=1` added to the container the same requests run
clean.

### A device-side trap resets the GPU when compute-wave save and restore is off

A kernel that executes a trap should end only its own process: a failed
`CUDA_KERNEL_ASSERT` in PyTorch (which ROCm builds as `abort()`) or a
`__builtin_trap()` makes the HIP runtime report `HSA_STATUS_ERROR_EXCEPTION`
and exit. With `amdgpu.cwsr_enable=0` the card is reset as well. The kernel logs
`SQ instruction intr detected ... trap true` and the process exits. About ten
seconds later come `qcm fence wait loop timeout expired`,
`unsuccessful queues preemption`, `Failed to evict process queues`, a BACO
reset and `VRAM is lost`, and every ROCm process on the card loses its work.

ROCm's trap handler for gfx8 (ROCr `amd_trap_handler_v1.h`) signals the error
to the runtime and then halts the wave (`s_sethalt 1`), so the wave never
finishes. When the process exits, the driver has the hardware scheduler
preempt its queues. A queue can be preempted only when its waves finish or are
saved by compute-wave save and restore (CWSR). With CWSR off, the scheduler
waits for the halted wave until the driver's 9-second preemption timeout runs
out. gfx8 has no per-queue reset, so the driver's only recovery is to reset
the whole GPU. With CWSR on, the halted wave is saved like any other, the queue
is removed and the card keeps running.

The same one-wave trap program, on the same kernel (6.18.57) and stock
runtime, reset the card on its first run with `cwsr_enable=0`. With CWSR on,
none of seven trap shapes reset it: one lane; every lane of 64 blocks; a trap
while the rest of its workgroup waits at a barrier; a trap while the other
blocks keep running for seconds; a trap while a second ROCm process keeps the
card busy; every lane on the patched runtime; and an out-of-range PyTorch
index in a process with rocBLAS loaded. Each ended with
`HSA_STATUS_ERROR_EXCEPTION` in the process, no driver message after the trap
line, and the other process finishing normally.

CWSR is the driver's default. Leave it on:
`/sys/module/amdgpu/parameters/cwsr_enable` should read 1. If it reads 0,
remove `options amdgpu cwsr_enable=0` from `/etc/modprobe.d/` (or
`amdgpu.cwsr_enable=0` from the kernel command line), rebuild the initramfs
and reboot. To check, run a HIP kernel in which one lane calls
`__builtin_trap()`. The process should report `HSA_STATUS_ERROR_EXCEPTION`,
and `journalctl -k` should show no `qcm fence` line after the trap.

## 4. The HSA runtime: AMD's stock build

The images run AMD's stock `libhsa-runtime64` (`.so.1.15.60404`) from the ROCm
base image. Schaka's gfx803 tree patches it with a "va-reuse-defer" change
to how freed GPU virtual addresses are reused
(`patches/rocr/va-reuse-defer.patch` in git history up to commit `0e67b7d`;
investigation in `KERNEL_BUGS.md`), written for MIOpen's CK reduction kernels
reading stale data when a freed address is reused at once (ONNX Runtime's
`ReduceSum` test). The images do not build it, for two reasons.

It page-faults on this card ("Page not present", writes from the texture
cache) in workloads that free and reallocate GPU memory in particular
patterns: a synthetic GEGLU op test (`gfx803_unet_check.py --only GEGLU`), and
wide GEMMs followed by a 4.9M-element transposed copy, every run, where the
stock runtime passes. Ollama's llama-server crashed on the same faults
mid-prompt (a 17k-token Qwen Code request on Qwen3.5 9B: ten
`VM fault ... write from 'TC*'` lines, then an interrupt ring overflow), and a
storm of such faults in several processes overflowed the interrupt ring and
froze the desktop until a reboot. The GEGLU test still faults with the CLR fix
below in place, so the faults are the patch's own. ComfyUI faulted twice on
the second generation after a model reload while it ran the patched runtime,
which fits the same pattern but is not proven.

And the bug it was written for is the stale vector L1 described next: with
CLR's cache invalidate, the MIOpen reduce programs pass on the stock runtime.

Profilers work on the stock runtime: `rocprof --hip-trace --stats`,
`rocprofv3 --kernel-trace --stats` and `torch.profiler` record kernels, with
or without the SGEMM shim preloaded.

### A kernel can read stale data from its CU's caches across dispatches

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

Cause: at a dispatch boundary the command processor of gfx803 (MEC firmware
730) leaves cache lines valid that ROCm's contract says it drops. ROCm
expresses cross-dispatch coherence only through the fence scopes in the AQL
packet header, and the LLVM memory model for GFX6 to GFX9 relies on the packet
processor to act on them, so compiled code never invalidates a cache itself.
Which lines survive depends on how they were fetched.
`tools/tc-staleness/loadpaths.hip` refills a recycled 4 KiB buffer 1,000 times,
reads it back each time through one load path, and empties both caches before
each path. With the packet described below switched off:

| Load path | Stale reads of 1,000 |
| --- | --- |
| `flat_load` (every global load the compiler generates for gfx8) | 0 |
| `s_load` (the compiler's scalar loads of kernel arguments and uniform data) | 0 |
| `buffer_load` through a descriptor with word3 bit 27 clear | 993 to 996 |
| `s_buffer_load` through such a descriptor | 989 |
| either buffer load with bit 27 set | 0 |

Bit 27 of a buffer descriptor's word3 is the low bit of the memory-type field
(MTYPE) in ROCr's gfx8 register layout; AMD's GCN3 ISA guide calls it NV. A line
that a buffer instruction fetched through a descriptor with the bit clear
survives the boundary, in the vector L1 and in the scalar cache alike. Once a
line survives, any later load of its address hits it, whatever path that load
takes. The compiler's own loads fetch no such lines. Its register spills do,
through the scratch descriptor ROCr builds with the bit clear, but every reload
follows the same wave's store. Hand-written assembly and code built on buffer
intrinsics fetch them all the time: Tensile's gfx803 assembly GEMM kernels,
MIOpen's assembly convolutions and its composable-kernel reductions all build
descriptors with the bit clear. That is why every failure in the table below goes through
rocBLAS's assembly kernels or MIOpen's reductions, and why probes through
PyTorch's own kernels stay clean. At kernel entry, `buffer_wbinvl1` cures the
vector case and `s_dcache_inv` the scalar one;
`buffer_wbinvl1_vol`, the invalidate LLVM emits for an acquire on gfx8, cures
neither.

A kernel on a CU whose caches still hold such lines of a page that something
else has rewritten since then reads the old bytes. The rewrite is a copy: copies
of 16 KiB or less are shader copies that write through L2, and an address comes
back either because the application refills the same buffer or because ROCr's
fragment heap hands a freed 4 KiB page to the next `hipMalloc` of the same size,
with no ioctl, unmap or TLB flush in between. Operands much larger than the 16
KiB L1 push the old lines out on their own, which is why big GEMMs were never
affected. The wrong values are what was last at that address (a zero-padded and
a NaN-padded buffer give different garbage at the same case, so this is not an
out-of-bounds read), serialising launches does not cure it, and neither do SDMA,
the SGEMM shim or the workspace size.

Fix: `patches/clr/` rebuilds `libamdhip64` from ROCm/clr at the rocm-6.4.4 tag
with one PM4 `ACQUIRE_MEM` packet, published in its own AQL ring slot ahead of
every kernel dispatch, that invalidates the vector L1 and the scalar cache
(`CP_COHER_CNTL` = `TCL1_ACTION_ENA | SH_KCACHE_ACTION_ENA`). Only the one
library is replaced; the Ollama image carries the identical stock file, so
the same rebuilt file fits it. `CLR_GFX8_TC_INVALIDATE=0` switches the packet off in the same
binary, which is how the fix is measured against itself.

The invalidate sits in the HIP runtime because the runtime issues the dispatch
packets whose fence scopes promise it. Two other placements were measured and
not taken. Setting bit 27 in every gfx8 buffer descriptor leaves no line to go
stale and costs nothing measurable inside a kernel (within 0.3% for re-reads of
4 KiB to 16 MiB). With Tensile's descriptors patched that way, the rocBLAS probe
passes all 727 cases with the packet off. But correctness would then depend on
no kernel anywhere building a descriptor without the bit, which nothing
enforces, and on undocumented firmware behaviour. A scan of every gfx803 code
object in the images finds two that build one: Tensile's assembly kernels and
the attention backward kernel in `tools/`. MIOpen builds more of them at run
time. An invalidate at every kernel's entry keeps the contract instead, and
`s_dcache_inv` there costs next to nothing except on very large grids (3.5 us on
an empty kernel of 16k waves). But it needs a compiler change, every library
rebuilt and the assembly kernels edited, and it leaves precompiled code
uncovered. Either would save only the packet's cost per dispatch (RT-1 in
`OPTIMIZATION_TARGETS.md`).

Measured on the RX 590 with the stock HSA runtime, the same binary and only
`CLR_GFX8_TC_INVALIDATE` toggled:

| Test | Packet off | Packet on |
| --- | --- | --- |
| rocBLAS probe, 511 free-and-reallocate cases | stops at case 225 | passes |
| `torch.linalg` through the rebuilt rocSOLVER, 204 cases up to n=257 | wrong | 204 of 204 |
| MIOpen reduce kernels, 52 shapes (`tools/reduce-harness`) | 12 fail | 0 fail |
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
clean on this stack with the packet off as well, because PyTorch's kernels load
through flat instructions. `loadpaths.hip` and the rocBLAS probe isolate the
bug, together with the rocBLAS and reduce programs in the table.
That MIOpen's reduce failures and rocBLAS's wrong results on small fp32 GEMMs
(`KERNEL_BUGS.md`) are this bug follows from the table, and the patches
written for them before its cause was known have nothing left to fix. On the
stock runtime with the packet, the MIOpen reduce programs pass every shape in
five runs each (10 and 11 of 52 shapes fail with it off), so the
va-reuse-defer runtime change is not needed. Every rocBLAS solution for fp32
strided-batched GEMMs of 1 to 16 cubed (batch 4), forced one at a time by
index on buffers refilled before each call, returns the exact product in 300
of 300 calls; with the packet off nearly all of them are wrong in most calls,
including the assembly kernels that the small-shape rocBLAS patch steers away
from, so its detour is not needed either; that rocBLAS patch is still in the
images. The faults of the patched HSA runtime have not been explained. The intermittent
training fault in section 10 has not recurred with the fix in 25 runs of the
setup it occurred in; whether this bug caused it is still open.

### Device printf and assert() hang: no-return atomics on host memory

A HIP kernel that calls `printf`, or an `assert()` that fails (it prints its
message the same way before trapping), never finishes on gfx803. Nothing is
printed, `hipDeviceSynchronize` never returns, the GPU stays 100% busy, and
killing the process is the only way out. PyTorch is not affected because its
ROCm build turns `CUDA_KERNEL_ASSERT` into `abort()`, which traps without
printing.

The cause is how atomics on host memory execute. On this RX 590 (Ryzen 7
2700X host, PCIe atomics enabled), a system-scope atomic read-modify-write on
fine-grained host memory runs as a plain store of its operand when the
instruction does not return the old value (`flat_atomic_add_x2` without
`glc`). Whether other Polaris hosts or newer GPUs behave the same is not
tested; the patch below changes code generation for gfx8 only. One such atomic applied to a
`hipHostMallocCoherent` word holding 1 gives these results:

| Operation | Correct result | Result on gfx803 |
| --- | --- | --- |
| add 5 | 6 | 5 |
| sub 1 | 0 | 1 |
| or 2 | 3 | 2 |
| and 3 | 1 | 3 |
| xor 3 | 2 | 3 |

A later GPU read sees the same values, so nothing is held back in L2. The
returning add and swap are correct. Plain `atomicAdd()` with its result unused
gives the same wrong value on `hipHostMalloc`'s default allocation, which is
fine-grained too. A no-return add is correct on `hipHostMallocNonCoherent`,
`hipMallocManaged`, fine-grained device memory (`hipExtMallocWithFlags`) and
`hipMalloc`.

HSA signals live in fine-grained host memory, and ockl's
`__ockl_hsa_signal_add` and its siblings discard the old value, so the compiler
picks the no-return instructions. Device printf goes through hostcall: the wave
pushes a packet onto a stack in host memory and rings a doorbell signal with
`__ockl_hsa_signal_add(doorbell, 1)`. The doorbell starts at 1, and 1 + 1 is
stored as 1. CLR's listener thread waits for the doorbell's value to change, so
it never looks at the packet. Read from the process while it hangs, the
hostcall buffer shows the packet on the ready stack with its ready bit set and
the doorbell still at its initial value.

[`ockl-gfx8-signal-returning-atomics-6.4.4.patch`](../patches/device-libs/ockl-gfx8-signal-returning-atomics-6.4.4.patch)
makes ockl's signal updates keep the old value on ISA versions below 9, which
selects the returning instruction. A Dockerfile stage builds `ockl.bc` from
the device-libs source at the commit the image's compiler was built from and
installs it in the final image. Built the same way without the patch, it
reproduces AMD's shipped `ockl.bc` byte for byte. With the patched bitcode,
printf from 1 and 64 lanes prints every line and returns `hipSuccess`, and a
failed `assert()` prints `Device-side assertion ... failed` and ends the
process. Code compiled with hipcc or clang in the final image links the new
bitcode, and so do kernels compiled at run time through hiprtc and MIOpen:
`libamd_comgr` embeds its own copy, and the image's `libamd_comgr` is built
from a tree carrying the same patch (next section). Libraries built in the
earlier stages keep the copy they were linked with.

One problem remains: a user kernel that applies a no-return atomic to
fine-grained host memory (including `atomicAdd()` on memory from a plain
`hipHostMalloc`) still gets the wrong value. Use the returned value, or keep
such counters in device memory. A compiler change that selects the returning
atomic for user code on gfx8 would fix it everywhere; the image's compiler
patches do not include one.

To check, build a kernel in which one lane calls `printf("x\n")` and
synchronize. Without the patched `ockl.bc` the program hangs; with it the line
prints and the call returns.

### Non-inlined device functions read implicit kernel arguments from address 0

Symptom: with AMD's compiler, a PyTorch binary elementwise op on an fp16 and an
fp32 tensor (`torch.maximum`, `atan2`, `fmod`, `hypot`, `logaddexp`,
`mse_loss`, `sigmoid_backward`, `gelu_backward` and others) ends the process
with `Memory access fault ... on address (nil)` whenever the element count is
not a multiple of 16384; at an exact multiple it is correct. A failed
`assert()` faults at its trap the same way.

Cause: on gfx8 a few values come from the kernel's implicit arguments that
gfx9 and newer read from hardware registers: the queue pointer the trap
handler expects in `s[0:1]`, and the private and shared aperture bases
needed to turn a stack or LDS address into a generic pointer. LLVM's
`SITargetLowering::loadImplicitKernelArgument` computes their address from
the kernel-argument pointer, which a function that is not inlined does not
have, and then uses the bare offset: the callee loads the queue pointer from
address 0xc8 and the apertures from 0xc0 and 0xc4. The kernel already passes
the implicit-argument pointer to such callees (in `s[8:9]`); only this
lowering ignores it. Upstream LLVM has the same code. It reaches only
pre-gfx9 targets with code object v5 or later, which the images use.
PyTorch's vectorized elementwise kernel runs its last, partial block through
a non-inlined `elementwise_kernel_helper` whose prologue builds a generic
pointer to its stack; that is the fault above. A scan of every gfx803 code
object in the images finds this load in 155 functions of `libtorch_hip.so`
(64 aperture loads, the rest on assert and unsupported-cast paths), on trap
paths only in llama.cpp, and nowhere in rocBLAS, rocSOLVER, rocSPARSE,
MIOpen's cached kernels, MIGraphX, ONNX Runtime or the SGEMM shim.

Fix (`patches/llvm/0001-*`): in a callee, take the implicit-argument pointer
the kernel passes. The image builds `clang-19` and `libamd_comgr` (which
embeds its own copy of the compiler for hipRTC and MIOpen's run-time
kernels) from ROCm llvm-project `rocm-6.4.4` with the patch, builds PyTorch
with that `clang-19`, and ships both. The callee then emits
`s_load_dwordx2 s[0:1], s[8:9], 0xc8` where AMD's compiler loads from a null
base. On the card: the 8 mixed-dtype ops above at 1000, 1052672 and 1048577
elements are correct in 24 of 24 runs (24 faults before), a non-inlined
stack-pointer cast passes, a failed `assert()` ends with
`HSA_STATUS_ERROR_EXCEPTION`, and the SDXL UNet output is bit-identical at
the same step time. The libraries built before PyTorch keep AMD's compiler;
none of their code objects has the null-base load. A failed device-side
assert now reaches `s_trap`, which with compute-wave save and restore off
resets the GPU (section 3).

A second gfx8 bug sits behind the first for LDS: a flat (generic-pointer)
access that lands in LDS is bounds-checked against M0, and LLVM sets M0 only
for direct LDS instructions. A kernel that reaches LDS only through a
generic pointer, as a non-inlined function returning a pointer into a
`__shared__` array makes it, runs with M0 unset: its stores are dropped and
its loads return 0, with no error. With the first fix alone such a callee
stops faulting and starts returning zeros. `patches/llvm/0002-*` selects
`s_mov_b32 m0, -1` for every generic access on gfx6-gfx8, as for direct LDS
accesses, and records the M0 use on the instruction so that later passes
keep the initialisation in front of it; the probe then reads back every
value. In one PyTorch source file it adds four `s_mov_b32` instructions to
7,484 flat accesses.

To check: `torch.maximum(torch.randn(1000, device="cuda").half(),
torch.randn(1000, device="cuda"))` returns the elementwise maximum; with
AMD's compiler it faults (with 1048576 elements it does not).

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
checks it. [`patches/pytorch/`](../patches/pytorch/) routes
it to rocBLAS whenever hipBLASLt is not the preferred backend, as PyTorch's
floating-point GEMMs already do. The fix comes from Schaka's ROCm 10 line
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

[`gfx803-layer-norm-warp-size.patch`](../patches/pytorch/gfx803-layer-norm-warp-size.patch)
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
With compute-wave save and restore turned off, that trap also resets the
card instead of only ending the process (see
[the trap section](#a-device-side-trap-resets-the-gpu-when-compute-wave-save-and-restore-is-off)). The scan
as written gets 100 of 100 random blocks wrong on the card, and 0 with the
hardware's warp width
([`tools/correctness-suite/pytorch/`](../tools/correctness-suite/pytorch/)
holds this reproduction and the checks quoted below).
[`gfx803-topk-scan-warp-size.patch`](../patches/pytorch/gfx803-topk-scan-warp-size.patch)
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
[`gfx803-rocprim-private-namespace.patch`](../patches/pytorch/gfx803-rocprim-private-namespace.patch)
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
[`gfx803-index-put-warp-size.patch`](../patches/pytorch/gfx803-index-put-warp-size.patch)
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
[`gfx803-embedding-backward-row-width.patch`](../patches/pytorch/gfx803-embedding-backward-row-width.patch)
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
image; the modules are in [`runtime/python/`](../runtime/python/).
They are workarounds with their source fixes still to do.

- **accelerate on a build without torch.distributed.** accelerate 1.15 imports
  `torch.distributed.tensor` unconditionally, which crashes on this
  single-GPU build; the image stubs the check (a model on such a build cannot
  hold distributed tensors).

## 6. ComfyUI node with gfx803 kernels for attention, GroupNorm and GEGLU

PyTorch has no fused attention on gfx803, so every SD 1.5 attention call
stores its full score matrix (8 x 4096 x 4096 floats at the 64x64 level).
[`tools/ComfyUI-gfx803-attention/`](../tools/ComfyUI-gfx803-attention/) is a
ComfyUI custom node that replaces three operations of ComfyUI's SD 1.5 and SDXL
UNets with HIP kernels from [`tools/`](../tools/):

- attention for head sizes 40, 64 and 80, in one pass with an online softmax
  (`gfx803_fused_attention.hip`: a block of four waves shares each 32-key K/V
  tile across 128 query rows), hooked into `F.scaled_dot_product_attention`
  and ComfyUI's split and optimized attention (fp16 inputs are computed in
  fp32); head size 160 and calls that need gradients stay on PyTorch;
- GroupNorm with float64 accumulation (`gfx803_group_norm_silu.hip`), fp32
  and up to 128 channels per group;
- GEGLU, the split, exact-erf GELU and product in one pass
  (`gfx803_geglu.hip`), fp32 like the GroupNorm: at ComfyUI's default fp16
  only the attention runs.

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
none) and copies the node into `ComfyUI/custom_nodes/`.

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
  12.4 -> 10.2 ms at the sustained clock; 5120x8192x640 13.9 -> 10.1 ms in
  a short timing at the boost clock). Bit-identical; fp16 UNet
  step SDXL 2.98 -> 2.84 s, SD 1.5 359 -> 355 ms. It takes fp32 TN calls
  too, with 8-deep slices (the same 32 bytes per row; 16-deep fp32 slices
  spill), 8-23% faster than the 64x64 tile on llama.cpp's prompt GEMMs and
  SD's fp32 linears. Its stores of an output whose columns lie a multiple
  of 32 KiB apart all hit one DRAM channel (8192x2048x2048 19.0 ms against
  17.6 on the 64x64 tile), so such fp32 calls keep the 64x64 tile, and
  llama.cpp patch 127 gives its 8192-row outputs padded columns (14.7 ms).
  Calls with too few 64x128 tiles run that kernel at 64x64, and at fp32 that
  form is slow (8-deep slices give it half the multiply-adds per barrier,
  and it fetches each row of A twice as often): 2.9-4.0 TFLOPS on
  llama.cpp's prompt matmuls where 64x128 reached 4.3-5.0 (4096x512x12288,
  a 512-token batch's down projection, 17.5 -> 10.3 ms). fp32 calls with
  whole 64x128 tiles take 64x128 there instead (bit-identical): Qwen3.5 9B's
  13,513-token prompt 63.1 -> 54.8 s, Qwen2.5-Coder 7B's 65.7 -> 59.1 s at
  512-token batches. fp16 keeps 64x64 there; its 16-deep slices ran within
  4% of 64x128.
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
(0.16-0.22) are within run-to-run variation. Over a full-length run (1200
steps at 512 px, the same dataset and seed) the two precisions track each
other step for step: the paired losses correlate at 1.0000, fp16's is lower by
0.0007 on average (0.0001 over the first 100 steps, 0.0013 over the last),
against a step-to-step spread of 0.10, and neither run has a non-finite loss.
The LoRA weights end as far apart (median per-tensor cosine 0.968, relative
difference 0.26) as one run's step-800 and step-1200 checkpoints are from each
other, and images rendered with the two at 1200 steps show the same subject
and style, for most prompt and seed pairs the same composition. fp16 costs no
measurable training quality. `gfx803-train` trains in fp16 mixed precision
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
slabs in a fixed order. At 4096 tokens, head size 40, it takes 17.6 ms against
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
to bf16's own rounding (about 2e-3), with no faults or hangs. Schaka's
ROCm 10 line reported bf16 hanging the card under vLLM (Schaka/rocm-gfx803,
`vllm/NOTES.md` there); that does not reproduce here. bf16 matmuls run at fp16/fp32
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
- **Image prompts 6x slower than text** (a 1080p image, 2061 tokens, 50.4 s
  against 8.8 s for 2028 text tokens, Qwen3.5 9B). 38.7 s of it was the
  vision encoder's attention at 0.21 TFLOPS. A vision encoder gives every
  query head its own K and V, and those tiles (ncols2 == 1, also used by
  text models with an odd GQA ratio) stayed on the half2 path at 64 columns,
  where they spill for every head size (305 VGPRs at head size 72, 817 at
  112). Moving them to the fp32 path made them slower; occupancy 1 at 64
  columns only halved the time. At most 32 columns (16 for head sizes 72,
  112, 128, 256), at occupancy 1 where it helps, they run 1.9-5.7x faster:
  the image 50.4 -> 18.7 s, and Qwen2.5-Coder 7B (GQA 7) prompts 8k deep
  60 -> 123 tok/s (patch 128). A gfx803 kernel for prompt-sized attention at
  head sizes 40-80 (patch 129), the layout of the ComfyUI node's multi-wave
  attention (a 4x4 block of scores per lane from registers, fp32, softmax
  weights passed by DPP inside the multiply-adds), runs the encoder's
  attention at 3.5 TFLOPS: 254 -> 88 ms per call, the image 18.6 -> 14.2 s,
  and Qwen3.6 35B-A3B through Ollama 14.0 -> 9.6 s. Loading the mask values
  inside the softmax spilled 200-590 VGPRs; loaded ahead of Q*K^T they fit.
- **Long prompts at head size 128 spent up to 44% of their time in attention**
  (Qwen2.5-Coder 7B, 13,316 tokens: 42.6 of 97.3 s). The tile kernel runs
  prompt attention at 2.0-2.5 TFLOPS where it serves the query heads of one
  K/V head together, and at 0.9 where the GQA ratio is odd (Coder 7B: 7), so
  it cannot. Patch 129's kernel does not fit there: 128 query rows of Q in
  fp32 take 67.6 KB of shared memory alone. Patch 130 keeps Q in registers
  instead, split over the eight lanes that share a set of query rows, and Q*K^T
  takes each lane's slice through the same DPP rotation that P*V uses for the
  softmax weights, so reading Q costs no instruction; blocks of 64 or 96 query
  rows, picked per batch. Prompt attention runs at 3.4-3.8 TFLOPS: Coder 7B's
  13k-token prompt 97.4 -> 65.8 s (attention 42.6 -> 10.7 s; through Ollama
  90.3 -> 56.5 s), Qwen3 8B's 78.3
  -> 74.6 s (attention 21.1 -> 17.1 s; at 512-token batches its 32 heads leave
  part of the last wave of blocks idle). At head size 96 the tile kernel's grouped tiles are 10-20x
  slower still. llama.cpp splits a head's keys over several blocks to fill the
  last wave; for this kernel the partial results and their combination cost
  more than that tail (a 2048-query batch over 512 keys: 2.8 ms unsplit, 9.4
  ms split), so it splits only when one block per output tile would leave over
  a quarter of the GPU idle. Why the partial-result stores are that slow is
  not understood: wider stores, another layout and half the writers changed
  nothing.
- **Prompt attention at head size 256** (the full-attention layers of
  Qwen3.5, Qwen3.6 and Qwen3-Coder-Next) ran on the tile kernel at 1.5-2.1
  TFLOPS, and at 0.1-0.3 where it serves one query head per tile. Patch
  130's kernel does not fit there: split over eight lanes, two query rows
  hold 64 floats of Q and 64 of output per lane, and a 32-key K/V tile pair
  takes 66.5 KB of shared memory. Patch 131 splits each set of query rows
  over a whole 16-lane row (16 floats of Q and of output per row and lane,
  DPP rotation over 16 lanes) with 16-key tiles (33 KB): 3.0-3.5 TFLOPS,
  Qwen3.5 9B's 13,513-token prompt 65.7 -> 62.5 s (attention 6.16 -> 3.57
  s). Against an fp32 reference its error is the fp16 rounding of K and V
  (3-4.5e-8 NMSE), the tile kernel's up to 4.8e-5. Checks against the CPU
  backend with f16 K and V measure distance to a reference that itself
  accumulates the output in fp16 and rounds Q to fp16.
- **Prompt attention wrote its output at a few GB/s.** Patch 130's kernel
  split a head's keys over blocks only where the GPU would otherwise idle,
  because the split's partial results cost far more than their size (a
  2048-query batch over 512 keys: 3.06 ms unsplit, 8.55 ms split), and even
  unsplit its stores took 0.67 ms for 16.8 MB. Consecutive query rows of a
  head lie n_head x D floats apart in the output, 8 KiB at 16 heads and head
  size 128, and a stride that size maps every row to the same DRAM channel;
  blocks start with the query tile fastest, so all blocks in flight wrote
  rows of one head into that channel. Writing the same rows head-major
  (consecutive lanes 512 B apart, wrong output, timing only) took 2.42 ms.
  `136-gcn4-fattn-prompt-head-order.patch` starts the heads of one K/V head
  side by side, whose rows lie D floats apart and who read the same K and
  V: 2.47 ms unsplit, 3.96 ms split, same bits. At long contexts a block's
  compute outweighs its stores, so real prompts gain little (Qwen3.6's
  head-256 attention 10.03 -> 9.76 ms per call at 8k depth).
- **A fully masked query row came out NaN when attention was split.** A
  kernel that splits a head's keys over blocks leaves the final division to
  `flash_attn_combine_results`, which divides the weighted partial results by
  the weighted sums; for a row whose keys are all masked every sum is 0, and
  the combine wrote 0/0 where the CPU backend writes 0. test-backend-ops
  showed it on patch 130's kernel (head 128, 113 keys, 512 queries over two
  sequences) inside the full suite only; with the split forced off the suite
  passed. Only padding rows are fully masked in a model's prompt, and their
  output is discarded. `138-fattn-combine-masked-row.patch` writes 0 for an
  empty sum; the combine is generic code, so other GPUs reach the same NaN.
- **Prompt batches that are not a multiple of 64 tokens missed the fast
  SGEMM tile.** The shim's 64x128 two-wavefront tile needs whole 128-column
  tiles, and the columns of a prompt matmul are the batch's tokens, so the
  last batch of a prompt (752 tokens of a 2,800-token prompt in 2,048-token
  batches) went to an older tile. `137-gcn4-prompt-gemm-pad-tokens.patch`
  rounds such a batch up to 128 columns when that adds at most a sixteenth
  and drops the zero columns afterwards: Qwen3 8B at 752 tokens 256 -> 267
  tok/s, at 1,000 tokens 245 -> 262, Qwen3.5 9B at 752 tokens 252.5 ->
  266.7; same bits.
- **fp16-weight matmuls accumulated in fp16** (the vision encoder of an
  image model, any F16 GGUF). llama.cpp runs a matmul with fp16 weights
  through cuBLAS in fp16 compute on every AMD GPU but CDNA and RDNA4;
  rocBLAS takes that on Tensile kernels that accumulate in fp16
  (`Cijk_Alik_Bljk_HB`): 2-9e-5 NMSE against the CPU at K = 1152-4304 (about
  1% RMS), against 4e-8 accumulated in fp32, at ~3.3 TFLOPS. gfx803 has no
  packed fp16 math, so fp16 accumulation gains it nothing. Patch 132 asks for
  fp32 compute on gfx803, which the base image's SGEMM shim runs on its fp16
  tiles: Qwen3.5 9B's vision encoder projections 1.3-1.6x faster, a 1080p
  image prompt 14.19 -> 13.61 s.
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
  37 ms of 41 hidden with SDMA). Dockerfile-Ollama sets it, together with
  `ROC_CPU_WAIT_FOR_SIGNAL=1`, which keeps dependency waits out of the SDMA
  queues (section 3). llama-bench
  pp2048 with all experts in RAM: 193 / 250 / 295 -> 243 / 321 / 367 tok/s at
  512 / 1024 / 2048-token batches. In Ollama with Chatterbox loaded beside
  it (3.2 GiB of the model on the card), prompts of 2k and 8k tokens went
  124 -> 144 and 122 -> 135 tok/s. Generation is unchanged. Logits are
  unchanged too (llama-perplexity with and without: PPL 12.1753, mean KLD
  0), and 767 GiB of byte-compared SDMA round trips next to running GEMMs
  found no mismatch. The scheduler is compiled into `libggml-base`, which
  `Dockerfile-Ollama` installs from its CPU build, so the patch is applied
  before that build (`patches/ollama/llama-cpp-sched/`) and the build checks
  the installed library for it. In the image, through Ollama with its own
  batch choice (from the context length and free VRAM, 512 to 2,048 tokens;
  see docs/STACK.md), a 3,100-token prompt reads 165 -> 196 tok/s and a
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
  without it). The reset is the SDMA engine hang in section 3; this patch's
  reproduction runs clean on kernel 6.18.57, and the image avoids that hang
  with `ROC_CPU_WAIT_FOR_SIGNAL=1`, but the patch has not been run with that
  setting. The rest of a
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
  same top token 100% (with SDMA on, the 1-token batches give the same),
  and Ollama's greedy output is identical. Unlike 109, it keeps the
  scheduler's synchronous copies and changes only what they wait for (an
  event instead of the whole stream); no SDMA fault occurred in any run.
  `GGML_SCHED_EARLY_HOST_INPUTS=0` disables it.
- **Every dispatch costs ~4.3 us on gfx803, and a MoE token issued ~1,425.**
  The command processor needs ~2.65 us per kernel launch, plus 1.65 us for
  the vector-L1 invalidate slot (section 4), however small the kernel; a
  Qwen3.6 35B-A3B token launched ~1,425 kernels, ~1,064 of them on the
  critical path, about 4.6 ms of a 39 ms token. Four fusions remove up to 300
  of them: the DeltaNet output and alpha projections take their following
  ADD into the matrix-vector kernel through the reshape between them
  (`113-mmv-fuse-add-through-reshape.patch`), the DeltaNet recurrence reads
  its 2 MiB state from the cache row instead of a copy (`114`), the one-token
  convolution step and its state update run as one kernel (`115`), and the
  end of each MoE layer (expert sum, shared-expert and residual adds, the
  next RMS norm and its q8_1 quantize) runs as one kernel (`116`). Each is
  taken only where its guards hold (one sequence, one token, no other writer
  of the bytes it reads late) and falls back to the separate kernels
  otherwise. llama-bench with every expert in RAM, tg128: 25.60 -> 27.51
  tok/s; through Ollama 22.7 -> 24.2 tok/s with prompt speed unchanged.
  Arithmetic is unchanged: greedy output byte-identical on four models, and
  llama-perplexity at 1-token batches gives the same PPL in every digit with
  KL divergence 0.
- **Decode kernels waited on latency, not bandwidth.** After those fusions a
  Qwen3.6 35B-A3B token runs ~730 kernels with ~10.2 ms of GPU time, and the
  small ones between the GPU splits and the CPU experts mostly wait. Every
  warp reduction in ggml (the norms, the q8_1 quantizer, the matrix-vector
  epilogues, the DeltaNet recurrence) and the MoE top-k's argmax did its
  xor-butterfly steps with `ds_bpermute`, a round trip to the LDS unit per
  step. `133-gcn4-dpp-butterfly.patch` takes them from DPP row rotations and
  quad permutations instead, which is exact because in a butterfly that
  starts at the widest offset the rotated lane already holds the xor
  partner's value (top-k 14.9 -> 12.2 us a call). The one-token DeltaNet
  recurrence kept two loads of its 2 MiB state in flight per wave and then
  waited out two reductions; with four columns per wave (`134`) a call takes
  23.4 us instead of 34.6 (about 180 GB/s). The fp32 router's matrix-vector
  loop waited for each load before issuing the next; unrolled on gfx803
  (`135`) the router takes 17.6 us instead of 20.5. Together about 0.8 ms of
  GPU time per token, all bit-identical: llama-perplexity's logits files
  compare equal on the MoE model, Qwen3 8B and Qwen3.5 9B, and Ollama's
  greedy output is byte-identical on three models. Generation through
  Ollama: medians 28.4 -> 29.2 tok/s over six generations each (the CPU
  expert split makes single runs vary by about 2 tok/s on a desktop).
- **The DeltaNet recurrence spent a wave per state column on prompts.** In a
  2048-token prompt batch of Qwen3.6 35B-A3B, GATED_DELTA_NET took about 44
  ms per call, 30 calls and 1.71 s of about 6.6 s of GPU kernel time (26%),
  for about 225 GFLOP, under 2% of the card's fp32 rate. The kernel gives
  each 128-float state column one wave and walks the tokens in order; per
  token every lane loads k, q, v, the gate and beta from global memory and
  the wave does two wave-wide reductions around a few multiply-adds per
  lane. `117-gdn-prefill-column-kernel.patch` runs batches of more than one
  token with head size 128 on a kernel in which two neighbouring lanes hold
  a column (64 rows each in registers) and finish each dot product with one
  DPP add, a block of 128 threads covers 64 columns, and the token vectors
  are staged in LDS in chunks of 16 tokens (8 with KDA). On gfx803 its token
  loop is v_mac_f32 and v_mul_f32 only, at 159-165 VGPRs with no scratch.
  llama-bench with every expert in RAM, pp2048 at 2048-token batches: 361
  -> 455 tok/s (+26%); single-token decode keeps the old kernel, so
  generation is unchanged. The 128-long sums are added in a different
  order, so outputs are not bit-identical: test-backend-ops passes 47/47
  GATED_DELTA_NET cases against the CPU at NMSE 1e-7 with either kernel,
  and against CPU logits (llama-perplexity, 512-token batches, q8_0 KV
  cache) mean KLD is 0.0107 +- 0.0005 against 0.0104 +- 0.0005 with the old
  kernel, PPL 13.384 against 13.381 (CPU 13.355), same top token 95.3%
  against 95.8% (+- 0.45). The recurrence carries rounding forward through
  every token, and both GPU kernels sit about 0.01 KLD from the CPU.
  Through Ollama a 2,800-token prompt reads 334 -> 401 tok/s (+20%); greedy
  output is unchanged on models without DeltaNet layers, and on Qwen3.6 it
  follows the old text for 220 characters and then diverges at a near-tie
  (an em dash against a comma).

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
([`ollama-tools/requant-output-head.py`](../ollama-tools/requant-output-head.py))
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
- **Load MoE models without mmap.** With `use_mmap false` the experts sit in
  pinned host memory and each batch's expert copies go straight to the copy
  engine. From a memory-mapped model file every copy first asks the kernel to
  pin the file's pages for the GPU; KFD registers user memory as writable, the
  read-only mapping refuses, the kernel logs `amdgpu: init_user_pages: Failed
  to get user pages: -1`, and the copy falls back to a staged path, one failed
  pin per 1 MiB chunk. With patch 108's prefetch that is about 6000 failed pins
  a second during a long prompt. Qwen3.6 35B-A3B on 13,227-token prompts:
  192 tok/s mapped, 229 tok/s with `use_mmap false`, generation 1.5-2% faster,
  the same output, no kernel messages. The cost is RAM that cannot be swapped
  or reclaimed: 16 GB of pinned experts for this model.
- **Threads: one per core.** 16 threads on the 8-core CPU dropped decode to
  2.7 tok/s.

Averaged over a whole token, generation moves about 15 GB/s from RAM, but the
CPU expert phase is not compute-bound: its 609.5 MB of expert weights per token
(Q4_K gate, up and down, Q6_K down in half the layers) are read in about 16.7 ms,
36.5 GB/s, about 90% of the practical DDR4-2800 read rate. The rest of the token
is GPU work and the roughly 80 GPU/CPU handoffs, the part the decode fusions
(patches 113-116, 120-122) shorten. Against the CPU backend the model's logits
differ by a KL divergence of 0.014 (same top token 93.9%), in line with dense models
that are known to be right on this stack (Qwen3 8B 0.010, Qwen3.5 9B 0.007).
It is the first local model here that completes Qwen Code's tool-calling
tasks.

On gfx803 the quantized-integer matmul kernel (mmq) is the slow path for MoE
prompts. Without an int8 dot instruction each multiply-add of mmq's int8 dot
products costs about three instructions, so the expert matmuls of a
2048-token Qwen3.6 35B-A3B batch ran at 1.6-2.1 TFLOPS and took 62% of the
batch's GPU time. Dequantizing the Q4_K/Q6_K weights into shared memory and
multiplying in fp32 (v_mac_f32, one per lane per clock) is faster and also more
accurate, because the activations are no longer rounded to 8 bits: patch 118
does that for prompt batches as one grouped GEMM per call, 1.56-2.65x per call,
pp2048 450 -> 547 tok/s, mean KL divergence against CPU logits 0.0097 against
mmq's 0.0107. Chips with a native dot instruction keep mmq.

Each patch header in `patches/ollama/` has the full measurements and
a reproduction.

## 10. Open questions

- Why the binary Winograd kernels fail on gfx803 (they are disabled, not
  fixed). Candidates: their persistent launch, or wrong cache/release fields in
  their converted code-object metadata on gfx8.
- Why the patched HSA runtime's va-reuse-defer change page-faults on gfx803
  (GEGLU op test, wide GEMMs then a transposed copy, llama-server mid-prompt).
  The images do not build it. The reproducers are
  `gfx803_unet_check.py --only GEGLU` and the wide-GEMM-then-transposed-copy
  sequence described in section 4, run against a `libhsa-runtime64` built
  from ROCR-Runtime `rocm-6.4.4` with `patches/rocr/va-reuse-defer.sh` (commit
  `0e67b7d`), with the desktop stopped (a fault storm can freeze it).
- Whether PyTorch's group_norm computes wrong weight and bias gradients on
  gfx9 for a spatial size of 1 and more than 128 rows:
  `GammaBeta1dBackwardCUDAKernel2` reduces a 32-wide tile with shuffles over
  `C10_WARP_SIZE` lanes, which upstream sets to 64 there (section 5). That
  would be an upstream bug; not tested, as there is no gfx9 card here.
- Why llama.cpp's SDWA dot product (patch 104) is slower than the compiler's
  code in the prompt kernels, for table-lookup operands and for 2-token decode.
- Whether the decode kernels (mmvq) beat the prompt kernels (mmq) for 9-16
  tokens on gfx803; llama.cpp switches at 8.
- fp16 LoRA training in ComfyUI (only short runs so far).
- An intermittent GPU memory fault in SD 1.5 LoRA training on AMD's stock HSA
  runtime. Two of sixteen runs of 120-300 steps (same image and data) died, at
  steps 32 and 74, with `Memory access fault ... Page not present` and amdgpu
  `VM fault ... read from 'TC'` lines, without a GPU reset; the loss was healthy up
  to that point and nothing was NaN. To reproduce, run 300-step
  `gfx803-train` jobs on the SD 1.5 dataset in the setup the faults occurred
  in: `--precision fp32` with PyTorch's attention (`GFX803_ATTN_TRAIN=0`),
  about 3.5 s/step at a 7.2-7.6 GB peak. Check `journalctl -k` for `VM fault`
  afterwards; expect to need several runs.
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
  section 4 would produce a stale upper half. With that fix, 25 300-step runs
  in the setup the faults occurred in completed without a fault (six
  earlier, and nineteen with no desktop on kernel 6.18.57: sixteen with
  compute-wave save and restore on and three with it off). At the old rate of 2
  in 16, 25 clean runs in a row happen about 4% of the time, so the fault is
  either gone or several times rarer. The faulting runs were on an older
  kernel (6.18.55), so the clean runs do not single out the fix as the cure. The
  default setup has changed since: `gfx803-train` now runs the fused attention
  kernels, and in it fourteen 300-step and two 1200-step runs (fp32 and fp16)
  completed without a fault. Those runs do not execute PyTorch's attention
  kernels, so they show that the default setup has not reproduced the fault,
  not that the fix cured it.
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
- [`training/gfx803_unet_check.py`](../training/gfx803_unet_check.py):
  the op-by-op and whole-UNet check against a float64 CPU reference.
