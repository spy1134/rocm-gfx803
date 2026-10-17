# rocm-gfx803: ROCm 6.4.4 for AMD Polaris (gfx803)

AMD removed gfx803 GPUs (Radeon RX 470, 480, 570, 580 and 590) from ROCm after
version 6.0. The software still builds for them, but much of the stack either
returns wrong results or runs far below what the hardware can do. This project
maintains a ROCm 6.4.4 stack for these cards, rebuilt from source with the
fixes below, and packages ComfyUI (Stable Diffusion 1.5 and SDXL), LoRA
training with sd-scripts, and Ollama on top of it.

Every fix is checked against a CPU reference, because the usual failure on this
hardware is a program that completes and returns wrong numbers.

## Results

Measured on an RX 590 8 GB. For images and training, "before" is an early setup
that already computed correct results (the original stack was faster in places,
but its results were wrong). For Ollama, it is the same llama.cpp version
without this project's patches.

| Workload | Before | Now | Speedup |
| --- | --- | --- | --- |
| SD 1.5 image, 20 steps at 512 px (ComfyUI, fp32) | 35 s | 10.5 s | 3.3x |
| SD 1.5 LoRA training step (4 images, 512 px) | 12.1 s | 3.4 s | 3.6x |
| Ollama prompt processing (Qwen3 8B, Q4_K_M) | 14 tokens/s | 223 tokens/s | 16x |
| A coding agent's first request (12,000-token prompt) | 18 min | 71 s | 15x |
| Prompt processing, MoE model with experts in system RAM, without and with expert prefetch (Qwen3.6 35B-A3B, 2048-token batches) | 295 tokens/s | 367 tokens/s | 1.2x |
| Ollama text generation (Qwen3 8B, short context) | 27 tokens/s | 30.5 tokens/s | 1.1x |

The largest gains come from replacing the code paths that were slowest on this
architecture:

- **GEMM.** A correct replacement for rocBLAS's fp32 matrix multiply, with
  register-tiled and pipelined kernels, reaches 3.9-4.8 TFLOPS on large
  matrices, up from 2.8-3.3. fp16 matrix multiplies in the layouts where rocBLAS
  is 15-25x slower now run on the shim's kernels.
- **Convolution.** A new Winograd F(2x2, 3x3) kernel replaces MIOpen's binary
  Winograd kernels, which corrupt memory on gfx803. It runs about 2x faster than
  the GEMM path for most SD 1.5 convolutions, in fp32 and fp16.
- **Attention.** An optional ComfyUI node fuses attention, GroupNorm and GEGLU,
  avoids a 2.3 GB score matrix, and takes a 512 px fp16 denoising step from
  460 to 393 ms (attention 161 -> 92 ms; SDXL self-attention 2.9x faster).
- **Language models.** Prompt processing uses fp32 GEMMs on the shim instead of
  emulated int8 dot products, and generation kernels were rewritten to avoid
  register spills. MoE prompts prefetch experts over the card's DMA engines
  while the GPU computes.

Text generation is limited mainly by memory bandwidth, so it had the least room
to improve. Its gain grows with context length (20.9 to 26.5 tokens/s at 8,000
tokens). Some changes show up as capacity rather than speed: fp16 training uses
1.7 GB less VRAM, which makes SDXL LoRA training fit on 8 GB, and an 8-bit KV
cache keeps a 16k-context Qwen3 8B entirely on the card.
[`docs/FINDINGS.md`](docs/FINDINGS.md) has the measurements behind each number,
and [`docs/OPTIMIZATION_TARGETS.md`](docs/OPTIMIZATION_TARGETS.md) lists what
could still get faster.

## Quick start

```
git clone https://github.com/spy1134/rocm-gfx803.git
cd rocm-gfx803/rocm6.4.4
./setup.sh                        # check the system and build the images
./start-comfyui.sh                # ComfyUI at http://localhost:8188
./start-ollama.sh run qwen3:8b    # chat with a local language model
```

`setup.sh` reports anything that is missing, asks which components to build,
and can be interrupted and resumed. The first run takes several hours because
the main libraries are compiled for gfx803. [`rocm6.4.4/README.md`](rocm6.4.4/README.md)
documents each component: the images, LoRA training, ComfyUI and Ollama.

## Requirements

- A gfx803 card. 8 GB of VRAM is recommended; training needs it.
- Linux with Docker and the buildx plugin, and your user in the `docker` group
  (`sudo usermod -aG docker $USER`, then log in again).
- About 100 GB of free disk space during the build. The finished images take
  about 31 GB.

## What was broken

Without this project's changes, on these cards:

- SD 1.5 LoRA training learned from wrong attention gradients (loss 3-4 instead
  of 0.1-0.2).
- MIOpen's binary Winograd convolutions corrupted memory and hung the GPU under
  sustained load, sometimes taking the machine down with it.
- MIOpen's BatchNorm training returned wrong statistics and gradients whenever
  a feature map's height times width was not a multiple of 4.
- PyTorch's `layer_norm`, `index_put`, `topk`, embedding backward, `nonzero`,
  boolean-mask indexing, `unique` and large sorts returned wrong results, failed
  to launch, or hung the GPU. The cause in several cases is one warp-size
  mismatch on 64-lane hardware.
- rocBLAS, rocSOLVER and `torch.linalg` returned wrong numbers at random: the
  card's vector L1 cache is not invalidated between kernel launches, so a
  kernel could read the old contents of a recycled buffer. rocSOLVER and
  rocSPARSE also shipped without any gfx803 code, so their kernels never ran.
- Ollama read prompts about ten times slower than the card allows, generated
  text below its memory bandwidth, and could fill VRAM until the desktop stalled.

All of these are fixed or mitigated. [`docs/FINDINGS.md`](docs/FINDINGS.md)
gives each problem's symptom, cause, fix and how to reproduce the check.

## Validation

- Fixes are compared against float64 CPU results operation by operation, and
  for whole models (the SD 1.5 UNet, Ollama's language models).
- Training was stress-tested for 45 minutes at full load, a run that previously
  crashed in five attempts out of five.
- All measurements are from one RX 590. Other Polaris cards share the chip
  design and should behave alike, but their clocks, memory and firmware differ,
  and none has been tested.

## Known issues

- The patched `libhsa-runtime64` from upstream page-faults in some workloads
  (it crashed Ollama mid-prompt), so the images preload AMD's stock runtime.
  `GFX803_RUNTIME=patched` selects the patched one for comparison. The cause is
  not yet understood, and ComfyUI's occasional fault on the second image after a
  model switch may be the same bug.
- A device-side assert on gfx803 does not raise an error. The GPU hangs and the
  driver resets it, which clears VRAM for every program using the card. A
  desktop usually survives; GPU programs must be restarted. The PyTorch bugs
  that triggered this are fixed, but a new one would look the same.
- Mining-edition cards often ship with memory clocked above its rating, which
  causes random faults and freezes. Check the memory clock first (see "Host
  VBIOS setting" in [`README-rocm10.md`](README-rocm10.md)).
- Linux kernels 6.12 and 6.13 have been reported to crash with these cards.

## Scope and origin

This is a fork of [Schaka/rocm-gfx803](https://github.com/Schaka/rocm-gfx803),
which built the original ROCm 6.4.4 setup and its first fixes. Upstream has
moved on to ROCm 10 and no longer maintains this version; this fork does. Only
`rocm6.4.4/` and `tools/` are maintained here. The rest of the repository (the
ROCm 10 build at the root and the archived `rocm7.14/`) is inherited from
upstream and left as it was; its documentation is
[`README-rocm10.md`](README-rocm10.md).

## Credit

The original 6.4.4 setup and its first fixes are [Schaka](https://github.com/Schaka)'s
(see [`rocm6.4.4/KERNEL_BUGS.md`](rocm6.4.4/KERNEL_BUGS.md)). Using ROCm 6.x with
libraries rebuilt for gfx803 comes from the community projects
[robertrosenbusch/gfx803_rocm](https://github.com/robertrosenbusch/gfx803_rocm) and
[woodrex83/ROCm-For-RX580](https://github.com/woodrex83/ROCm-For-RX580).
