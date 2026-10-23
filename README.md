# rocm-gfx803: ROCm 6.4.4 for AMD Polaris (gfx803)

Run Stable Diffusion, LoRA training and local language models on Radeon RX 470,
480, 570, 580 and 590 cards, with results you can trust.

AMD dropped these cards (gfx803, "Polaris") from ROCm after version 6.0. The
software still builds for them, but much of it returns wrong numbers or runs far
below what the hardware can do. This project rebuilds a ROCm 6.4.4 stack for
Polaris with those problems fixed, and packages ready-to-run tools on top of it.
Every fix is checked against a CPU reference, because the usual failure on this
hardware is a program that finishes and returns wrong results.

## What you get

- **ComfyUI** for Stable Diffusion 1.5 and SDXL image generation, with an
  optional gfx803 attention node that makes it faster.
- **LoRA training** for SD 1.5 and SDXL with sd-scripts, in 8 GB of VRAM.
- **Ollama** for local language models: dense and mixture-of-experts (MoE)
  models, long prompts, and image input, with the MoE experts in system RAM
  when the model does not fit on the card.
- **The base image** (`rocm-gfx803:rocm6.4.4`): PyTorch 2.8, MIGraphX and ONNX
  Runtime for gfx803, for your own projects.

## Requirements

- A gfx803 card: RX 470, 480, 570, 580 or 590. The smaller RX 460, 550 and 560
  and the R9 Fury/Nano generation share the instruction set but are untested.
  8 GB of VRAM is recommended; training needs it.
- Linux with Docker and the buildx plugin, and your user in the `docker` group
  (`sudo usermod -aG docker $USER`, then log in again).
- About 100 GB of free disk space during the build. The finished images take
  about 31 GB.

## Quick start

```
git clone https://github.com/spy1134/rocm-gfx803.git
cd rocm-gfx803
./setup.sh                        # check the system and build the images
./start-comfyui.sh                # ComfyUI at http://localhost:8188
./start-ollama.sh run qwen3:8b    # chat with a local language model
```

`setup.sh` reports anything that is missing, asks which components to build,
and can be interrupted and resumed. The first build takes several hours, because
the main libraries are compiled for gfx803; later rebuilds reuse that work.
[`docs/STACK.md`](docs/STACK.md) covers each component in detail: building,
running, training, models that fit in 8 GB, and troubleshooting.

## What to expect

Measured on an RX 590 8 GB with the current images.

| Task | Time or speed |
| --- | --- |
| SD 1.5 image, 512 px, 20 steps (ComfyUI, fp32) | 10.5 s |
| SDXL, 1024 px (fp16) | about 2.7 s per step |
| SD 1.5 LoRA training, 4 images at 512 px (fp16) | 2.5 s per step |
| Qwen3 8B, 13,300-token prompt, then generation (Ollama) | 66 s, then 25 tokens/s |
| Qwen2.5-Coder 7B, 13,300-token prompt, then generation | 56 s, then 29 tokens/s |
| Qwen3.5 9B, 13,500-token prompt, then generation | 55 s, then 31 tokens/s |
| Qwen3.6 35B-A3B (MoE, experts in system RAM), 13,500-token prompt, then generation | 26 s, then 25 tokens/s |
| A 1080p image as a prompt (Qwen3.5 9B / Qwen3.6 35B-A3B) | 13.4 s / 9.4 s |

Against the same software without this project's fixes:

| Workload | Before | Now | Speedup |
| --- | --- | --- | --- |
| SD 1.5 image, 20 steps at 512 px (ComfyUI, fp32) | 35 s | 10.5 s | 3.3x |
| SD 1.5 LoRA training step (fp16 now, fp32 before) | 12.1 s | 2.5 s | 4.8x |
| Prompt processing, 512 tokens (Qwen3 8B, Q4_K_M) | 14 tokens/s | 223 tokens/s | 16x |
| A coding agent's first request (12,000-token prompt) | 18 min | 71 s | 15x |
| A 1080p image as a prompt (Qwen3.6 35B-A3B) | 51.7 s | 9.4 s | 5.5x |
| Text generation (Qwen3 8B, short context) | 27 tokens/s | 30.5 tokens/s | 1.1x |

For images and training, "before" is an early setup that already computed
correct results; the original stack was faster in places, but its results were
wrong. Text generation is limited mainly by memory bandwidth, so it had the
least room to improve. [`docs/OPTIMIZATION_TARGETS.md`](docs/OPTIMIZATION_TARGETS.md)
has the measurements behind each change and what could still get faster.

### Where the speed comes from

- **Matrix multiply.** A correct replacement for rocBLAS's fp32 GEMM, with
  register-tiled and pipelined kernels, reaches 3.9-5 TFLOPS on large matrices.
  fp16 GEMMs that rocBLAS runs slowly, or would sum in fp16, run on the same
  kernels with fp32 sums.
- **Convolution.** A new Winograd kernel replaces MIOpen's, which corrupts
  memory on gfx803, and runs about 2x faster than the GEMM path for most SD 1.5
  convolutions.
- **Attention.** Fused attention kernels for ComfyUI, training and llama.cpp
  compute from registers in fp32, without the large score matrices.
- **Language models.** Prompts use fp32 GEMMs instead of emulated int8 dot
  products; MoE models copy their experts to the card in the background while
  it computes; generation kernels were rewritten to avoid register spills.

## What this fixes

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
- PyTorch's binary ops on an fp16 and an fp32 tensor (`torch.maximum`, `mse_loss`,
  `gelu_backward` and others) crashed whenever the element count was not a
  multiple of 16384, and a failed device `assert()` faulted instead of reporting:
  AMD's compiler reads some of a function's implicit arguments from address 0
  on gfx8 when the function is not inlined.
- Ollama read prompts about ten times slower than the card allows, generated
  text below its memory bandwidth, could fill VRAM until the desktop stalled,
  and ran fp16-weight matmuls (vision encoders) with about 1% error.

All of these are fixed or mitigated. [`docs/FINDINGS.md`](docs/FINDINGS.md)
gives each problem's symptom, cause, fix and how to reproduce the check.

## Validation

- Fixes are compared against float64 CPU results operation by operation, and
  for whole models (the SD 1.5 UNet, Ollama's language models).
- Training was stress-tested for 45 minutes at full load, a run that previously
  crashed in five attempts out of five.
- `tools/imgvalidate.sh <image>` checks a built image on the card in about 15
  minutes, including a control run that must reproduce a known bug with its fix
  switched off.
- All measurements are from one RX 590. Other Polaris cards share the chip
  design and should behave alike, but their clocks, memory and firmware differ,
  and none has been tested.

## Known issues

- **Mining-edition cards** often ship with memory clocked above its rating,
  which causes random faults and freezes that look like software bugs. Check
  the memory clock first (see "Mining-tuned VRAM clocks" in
  [`docs/STACK.md`](docs/STACK.md)).
- **Linux kernels 6.12 and 6.13** have been reported to crash with these cards.
- **Keep the driver's compute-wave save and restore on** (`amdgpu.cwsr_enable`,
  default 1; check `/sys/module/amdgpu/parameters/cwsr_enable`). With it off, a
  device-side assert does not just end the program that hit it: the driver
  resets the GPU, which clears VRAM for every program using the card (see "A
  device-side trap resets the GPU" in [`docs/FINDINGS.md`](docs/FINDINGS.md)).
- **If you write your own HIP code:** an atomic on fine-grained host memory (a
  plain `hipHostMalloc`, or `hipHostMallocCoherent`) whose result the kernel
  does not use stores its operand instead of updating the value (`atomicAdd()`
  of 5 to 1 leaves 5). Use the returned value, or keep such counters in device
  memory.

## Documentation

- [`docs/STACK.md`](docs/STACK.md): building and running the images, ComfyUI,
  LoRA training, Ollama and fitting models into 8 GB, every patch in the
  images, and host-side caveats.
- [`docs/FINDINGS.md`](docs/FINDINGS.md): what breaks on gfx803, why, and how
  each fix was checked, with steps to reproduce.
- [`docs/OPTIMIZATION_TARGETS.md`](docs/OPTIMIZATION_TARGETS.md): where the
  time goes, what has been measured, and what could still get faster.
- [`docs/KERNEL_BUGS.md`](docs/KERNEL_BUGS.md): the earlier kernel-level
  investigations, and a method for finding more.

| Path | What it holds |
| --- | --- |
| `Dockerfile`, `Dockerfile-ComfyUI`, `Dockerfile-Ollama` | the base image (ROCm 6.4.4 with the gfx803 libraries, PyTorch and the SGEMM shim), ComfyUI with training, and Ollama |
| `setup.sh`, `start-*.sh`, `container-args.sh` | the setup check and the launchers |
| `patches/` | every source patch the images apply, each with a header that says why |
| `runtime/`, `training/` | Python workarounds installed into the base image; the training tools in the ComfyUI image |
| `tools/` | correctness suite, the ComfyUI attention node, host setup, hang capture and repro tools |

## Origin

This project started as a fork of
[Schaka/rocm-gfx803](https://github.com/Schaka/rocm-gfx803), which built the
original ROCm 6.4.4 setup and its first fixes and has since moved on to ROCm 10.
It is now a standalone repository that maintains the ROCm 6.4.4 stack only; the
git history keeps all of the original commits. Work on another ROCm version
would go on its own branch.

## Credit

- [Schaka](https://github.com/Schaka) built the original ROCm 6.4.4 setup this
  project grew from: the base image with MIGraphX and ONNX Runtime for gfx803,
  the first rocBLAS and MIOpen fixes and the first SGEMM shim, the correctness
  suite, the vector-L1 staleness probes, the host setup notes and the
  investigations in [`docs/KERNEL_BUGS.md`](docs/KERNEL_BUGS.md).
- [robertrosenbusch/gfx803_rocm](https://github.com/robertrosenbusch/gfx803_rocm)
  and [woodrex83/ROCm-For-RX580](https://github.com/woodrex83/ROCm-For-RX580)
  showed that ROCm 6.x with libraries rebuilt for gfx803 works, and which
  settings Polaris needs; gfx803_rocm is also the source of the torch 2.6
  hardware validation and the kernel-version warning.
- The root-cause analysis in [ROCm/clr#269](https://github.com/ROCm/clr/issues/269)
  and the patched runtime in
  [lucbruni-amd/TheRock](https://github.com/lucbruni-amd/TheRock/tree/lb/gfx803-polaris-support)
  explain why ROCm 7 drops Polaris at the runtime level, which is why this
  stack stays on 6.4.4.
- Built on [ComfyUI](https://github.com/comfyanonymous/ComfyUI),
  [sd-scripts](https://github.com/kohya-ss/sd-scripts),
  [Ollama](https://github.com/ollama/ollama),
  [llama.cpp](https://github.com/ggml-org/llama.cpp),
  [PyTorch](https://github.com/pytorch/pytorch) and AMD's
  [ROCm](https://github.com/ROCm) libraries.
