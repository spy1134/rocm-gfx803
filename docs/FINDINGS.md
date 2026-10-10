# Findings: what breaks on gfx803, why, and what this fork does about it

This document collects what this fork found while making SD 1.5 training,
ComfyUI and Ollama work on a Polaris card. Each section states the symptom,
the cause as far as it is known, the fix or mitigation and where it lives, and
how to check it. All measurements are from one RX 590 8 GB (Polaris 20, 36 CUs)
on a Ryzen 7 2700X host, Arch Linux with the LTS kernel, ROCm 6.4.4 images
built from `rocm6.4.4/`. Upstream's earlier findings (the rocBLAS Tensile
bugs, MIOpen's grouped-conv out-of-bounds read, the ROCr allocator fix and
more) are in [`rocm6.4.4/KERNEL_BUGS.md`](../rocm6.4.4/KERNEL_BUGS.md) and not
repeated here.

A recurring lesson first: on this hardware the common failure is a correct-
looking run with wrong numbers, not a crash. Every fix below was verified
against a CPU reference, and "it runs" was never taken as evidence.

| Problem | Status | Where |
| --- | --- | --- |
| Attention gradients wrong (shim missed batched GEMMs) | Fixed | `patches/sgemm-shim-batched/` |
| Shim GEMMs 20x slow at power-of-two sizes | Fixed | same |
| Shim large GEMMs at 2.5-3 TFLOPS | Faster (3.8-4.8 TFLOPS, pipelined tiles) | same |
| Binary Winograd convs corrupt memory and hang the GPU | Fixed (solvers disabled) | `patches/miopen/winograd-bin-gfx803-disable.patch` |
| No fast 3x3 convolution after that | Fixed (new solver) | `patches/miopen/winograd-f2x3-hip-gfx803.patch` |
| fp16 3x3 convolutions slow (GEMM only) | Fixed (same solver, fp16) | same |
| rocBLAS fp16 N/T GEMMs 15-25x slow | Fixed (shim fp16 route) | `patches/sgemm-shim-batched/` |
| bf16 reported as supported (no bf16 hardware) | Checked: correct; convs 1.7x slower than fp16 | none needed |
| PyTorch int8 matmul fails on every call | Fixed (port of upstream's fix) | `patches/pytorch/` |
| PyTorch layer_norm wrong (warp-size mismatch) | Fixed | `patches/pytorch/` |
| torchvision nms has no gfx803 kernel | Worked around (CPU) | `runtime/python/` |
| Patched HSA runtime page faults | Open; images default to the stock runtime | `Dockerfile`, start scripts |
| VRAM exhaustion freezes the desktop | Mitigated (VRAM cap) | `training/gfx803-train.sh` |
| llama.cpp prompt processing 10x slow | Fixed | `patches/ollama/llama-cpp/100-*` |
| llama.cpp long-context prompts slow | Fixed | `patches/ollama/llama-cpp/101-*` |
| llama.cpp q8_0 generation slows with context | Fixed | `patches/ollama/llama-cpp/102-*` |
| llama.cpp K-quant generation below memory bandwidth | Fixed | `patches/ollama/llama-cpp/103-*` |
| Ollama rejects argument-less tools | Fixed (all GPUs) | `patches/ollama/ollama/` |
| llama.cpp reserves a vision projector's memory on the wrong device | Fixed (all GPUs) | `patches/ollama/llama-cpp-server/` |
| Ollama keeps stale free-memory figures on slow-starting GPUs | Fixed (all GPUs) | `patches/ollama/ollama/` |
| Ollama fills VRAM beside a desktop | Mitigated (margin from amdgpu's counter) | `start-ollama.sh` |

Paths in the table are relative to `rocm6.4.4/`.

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

fp16 is covered too; see section 7a.

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
validated on it. `GFX803_RUNTIME=patched` (start scripts, the app's manager)
and `gfx803-train --runtime patched` load the patched one for comparison. The
cost is the bug the patch was written for: MIOpen's CK reduction kernels can
read stale data when freed addresses are reused at once (ONNX Runtime's
`ReduceSum` test), a path PyTorch, sd-scripts and llama.cpp do not take. What
in the va-reuse-defer change faults on gfx803 is not investigated yet.

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
full-wavefront shuffles). The image used to route GPU layer_norm through
group_norm with one group; that route is gone.

The workarounds below are loaded into every Python process through `sitecustomize` in the base
image; the modules are in [`rocm6.4.4/runtime/python/`](../rocm6.4.4/runtime/python/).
They are workarounds with their source fixes still to do.

- **torchvision nms.** The compiled op has no gfx803 code object ("invalid
  device function"); it runs on the CPU.
- **accelerate on a build without torch.distributed.** accelerate 1.15 imports
  `torch.distributed.tensor` unconditionally, which crashes on this
  single-GPU build; the image stubs the check (a model on such a build cannot
  hold distributed tensors).

## 6. VRAM on an 8 GB card that also drives a display

Uncapped, PyTorch keeps freed memory and grows until the card is full: an SD
1.5 run that needs about 6.1 GB took 7.99 of 8 GB. The driver then moves
buffers to system memory and back (steps slowed from 4.3 to 60 seconds), and
the compositor fails to place its framebuffers ("Failed to pin framebuffer with
error -12"), which can freeze a monitor until a reboot. `gfx803-train` caps
PyTorch at what the card's other users hold at startup plus 1 GB for the HIP
runtime, measured from amdgpu's own VRAM counter (ROCm's count misses the
desktop's framebuffers). Even capped at 90% training caused brief pin failures
on a busy desktop, and 512 px SD 1.5 training does not fit beside a typical
desktop at all, so the app has a headless mode that stops the display manager.

## 7a. fp16, and what it makes possible (SDXL LoRA training)

**Correctness.** fp16 computes correctly on this stack: 63 operation cases
(every GEMM layout, batched, SD1.5's convolutions forward and backward data,
norms, softmax, attention) match a float64 CPU result from the same fp16
inputs to 2-9e-4, and rocBLAS accumulates fp16 GEMMs in fp32. The SD1.5 and
SDXL UNets in fp16 match fp32 to about 1e-3.

**Two slow paths, both fixed.**

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

**Training.** SD1.5 LoRA, 512 px, 20 steps: fp32 4.10 s/step at a 7.6 GB card
peak; fp16 4.40 s/step at 5.9 GB, same loss. fp16 is no faster (gfx803 runs
fp16 arithmetic at the fp32 rate), but it frees about 1.7 GB, enough for 512 px
training beside a desktop. Gradient checkpointing still cannot be turned off
at 512 px in either precision.

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

## 7. llama.cpp and Ollama

Ollama 0.35.1 builds llama.cpp b11232; its HIP backend compiles for gfx803
with no source changes, and computes correctly (llama.cpp's own operation tests
pass against its CPU backend). It was slow, for three gfx803-specific reasons
that share a theme: kernels tuned for GPUs with more instructions spill
registers, or run out of instruction throughput, on gfx803.

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


Each patch header in `rocm6.4.4/patches/ollama/` has the full measurements and
a reproduction.

## 8. Open questions

- Why the binary Winograd kernels fail on gfx803 (they are disabled, not
  fixed). Candidates: their persistent launch, or wrong cache/release fields in
  their converted code-object metadata on gfx8.
- Why the patched HSA runtime's va-reuse-defer change page-faults on gfx803
  (GEGLU op test, wide GEMMs then a transposed copy, llama-server mid-prompt),
  and whether it causes ComfyUI's faults after a model reload. The images run
  AMD's stock runtime until this is understood; the reproducers are
  `gfx803_unet_check.py --only GEGLU` and `workspace/scratch/wide_test.py`, on
  `GFX803_RUNTIME=patched` with the desktop stopped (a fault storm can freeze it).
- torchvision's nms op needs a source fix or gfx803 kernel target in the PyTorch / torchvision build.
- Whether PyTorch's group_norm computes wrong weight and bias gradients on
  gfx9 for a spatial size of 1 and more than 128 rows:
  `GammaBeta1dBackwardCUDAKernel2` reduces a 32-wide tile with shuffles over
  `C10_WARP_SIZE` lanes, which upstream sets to 64 there (section 5). That
  would be an upstream bug; not tested, as there is no gfx9 card here.
- fp16 over full-length training runs and in ComfyUI (only short runs so far).
- The `init_user_pages: Failed to get user pages: -1` kernel messages on every
  Ollama model load.

## Tools

- [`tools/gpu-hang-capture/`](../tools/gpu-hang-capture/): a 2-second GPU
  monitor, a watcher that captures a stuck compute queue's registers and
  in-flight dispatch packets within seconds (before the driver's reset erases
  them), and a decoder that names the stuck kernel. This is how the Winograd
  kernels were identified. Never read `/sys/kernel/debug/kfd/mqds` while queues
  are active: on this kernel it oopses and turns a GPU hang into a host lockup.
- [`tools/console-mouse/`](../tools/console-mouse/): mouse support for terminal
  apps on the Linux text console, through gpm.
- [`rocm6.4.4/training/gfx803_unet_check.py`](../rocm6.4.4/training/gfx803_unet_check.py):
  the op-by-op and whole-UNet check against a float64 CPU reference.
