# The ROCm 6.4.4 line for Polaris / gfx803

> New here? The [Quick Start on the front page](../README.md#quick-start) gets you running with
> `./setup.sh`; this file is the detailed reference.

Docker images for Polaris cards -- RX 460, 470, 480, 550, 560, 570, 580, 590,
plus the R9 Fury/Nano generation that shares the gfx803 ISA:

- `Dockerfile` -> `rocm-gfx803:rocm6.4.4`, the base: ROCm 6.4.4 with rocBLAS,
  MIOpen, MIGraphX, PyTorch 2.8 and ONNX Runtime built for gfx803, and the
  SGEMM shim.
- `Dockerfile-ComfyUI` -> ComfyUI, sd-scripts and the training tools
  ([below](#comfyui-and-sd-15-lora-training)).
- `Dockerfile-Ollama` -> Ollama with llama.cpp for gfx803
  ([below](#ollama-local-llms)).

The repository's [front page](../README.md) says where this line comes from and
what this fork changed; [`docs/FINDINGS.md`](../docs/FINDINGS.md) has the
detail behind each fix. This file is the reference for building and using the
images.

> **gfx803 is not a supported ROCm target.** AMD supports gfx900 and newer; on
> gfx803 everything here is best-effort and slow by construction (see
> [Expectations](#expectations)). This line is validated on an RX 590 8 GB:
> fp32 PyTorch (op by op and whole SD 1.5 UNet against a CPU reference), SD 1.5
> LoRA training, ComfyUI txt2img and Ollama. Upstream originally verified the
> base on a Sapphire RX 470 8GB with [`verify.py`](#verifying-on-hardware)
> (MIGraphX EP inference, rocBLAS GEMM, MIOpen convolution); its kernel-level
> fixes are described in [`KERNEL_BUGS.md`](KERNEL_BUGS.md), and the [fused-conv
> case study](#fixed-bug-case-study-rocmexecutionprovider-fused-conv-crash-musicnn-class-models)
> below shows one of them end to end.

## Why ROCm 6.4.4

ROCm 7 removed Polaris support outright. ROCR-Runtime rejects agents reporting
"legacy" HSA doorbell types 0 and 1, which is exactly what Polaris hardware
reports, so the GPU agent is dropped before it is ever enumerated --
`rocminfo` fails with `HSA_STATUS_ERROR` and `clinfo` reports zero devices. See
[ROCm/clr#269](https://github.com/ROCm/clr/issues/269), where an AMD engineer
root-causes it and publishes a patched ROCR in a
[TheRock fork](https://github.com/lucbruni-amd/TheRock/tree/lb/gfx803-polaris-support).

That patch restores enumeration, but the same commit's build config has to
exclude MIOpen, composable_kernel, hipBLASLt, hipSPARSELt, rocWMMA and
rocprofiler-compute for gfx803 -- which is most of what MIGraphX needs. On
ROCm 7, a Polaris image would enumerate the card and then have nothing to
dispatch convolutions to.

ROCm 6.x has no doorbell check at all. A stock `rocm/dev-ubuntu-24.04:6.4.x`
image enumerates a Polaris card as shipped, no runtime patch required, and
MIOpen still works. So this line pins the last ROCm 6 release rather than
carrying a patched runtime forward. (Upstream's ROCm 10 line, documented in
[`README-rocm10.md`](../README-rocm10.md), took the other road: it rebuilds the
whole stack from source with the doorbell and other runtime patches.)

## What gets rebuilt, and why

| Component | Source | Reason |
| --- | --- | --- |
| rocBLAS | `rocm-6.4.4` tag | Dropped gfx803 from its default `TARGET_LIST` at ROCm 6.0, but the gfx803 Tensile logic (`Logic/asm_full/r9nano/*.yaml`) is still in the tree, so `rmake.py -a gfx803` builds it back. The base image's prebuilt copy has no gfx803 code objects. |
| MIGraphX | `release/rocm-rel-6.4` | Prebuilt for gfx900+ only. |
| PyTorch | `ROCm/pytorch` `release/2.8` | No gfx803 wheel has ever been published. torchvision and torchaudio are built alongside it. `patches/pytorch/` routes int8 matmul to rocBLAS, since its default hipBLASLt path has no gfx803 kernels. |
| ONNX Runtime | `v1.22.2` | Built against the MIGraphX above. |

**MIOpen is built from source** at tag `rocm-6.4.4` with the patches in
[`patches/miopen/`](patches/miopen/). It still carries its
`Ellesmere`/`Baffin`/`Polaris10`/`Polaris11` → `gfx803` device map and its gfx803
solvers, which is the single thing that is dead on ROCm 7 and alive here, and
why the 6.x route is viable at all. It has no gfx803 `.kdb` (tuning database),
so it runs untuned.

Not every gfx803 solver it carries is usable. Its two binary Winograd solvers,
`ConvBinWinogradRxS` and `ConvBinWinograd3x3U`, are MIOpen's first choice for
nearly every fp32 3x3 convolution, and on gfx803 they corrupt memory the
convolution wrote and are the stuck dispatch in GPU hangs under sustained load.
[`patches/miopen/winograd-bin-gfx803-disable.patch`](patches/miopen/winograd-bin-gfx803-disable.patch)
makes both not applicable on gfx803, so those convolutions run on MIOpen's GEMM
solvers; its header has the measurements and a reproduction.

[`patches/miopen/winograd-f2x3-hip-gfx803.patch`](patches/miopen/winograd-f2x3-hip-gfx803.patch)
adds a Winograd solver back for gfx8: `ConvHipWinogradF2x3`, F(2x2,3x3) for 3x3,
stride-1, pad-1 fp32 convolutions, forward and backward data. It runs three
small HIP transform kernels around one strided-batched GEMM, which the SGEMM
shim below executes, and Find chooses between it and the GEMM solvers per shape.
On an RX 590 it runs most SD1.5 UNet convolutions about 2x faster, with lower
rounding error than the GEMM path, and SD1.5 LoRA training steps take 10-15%
less time. MIOpen keeps using find-db records made before the solver existed, so
delete the user `*.ufdb.txt` files once after installing a build with it
(`miopen/config/` for `start-comfyui.sh`).

## The SGEMM shim

rocBLAS/Tensile's fp32 GEMMs return wrong results on gfx803. The image builds
[`patches/sgemm-shim-batched/`](patches/sgemm-shim-batched/) in its own stage
and preloads it for every process (`LD_PRELOAD` in the final stage), so every
image built on this one inherits it. It replaces `rocblas_sgemm`,
`rocblas_gemm_ex` and their strided-batched forms (which attention uses) for
standard-algo fp32 calls, with register-tiled kernels that return correct
results and run large GEMMs at 3.8-4.8 TFLOPS on an RX 590, 3-4x stock
rocBLAS's speed. MIOpen's GEMM-based convolutions and the GEMM inside its gfx803 Winograd
solver also run through it. The original shim in
[`patches/rocblas/sgemm-shim/`](patches/rocblas/sgemm-shim/) is still built, at
`/opt/rocm/lib/libgfx803_sgemm_shim.so`, for comparisons only: it does not
replace the strided-batched calls, so attention gradients are wrong with it.
`GFX803_SGEMM_SHIM_DISABLE=1` passes every call to stock rocBLAS.

fp16 GEMMs are correct on rocBLAS's own kernels, but one layout (op(A) = N,
op(B) = T) runs 15-25x slower than the others there, and it is the backward
pass of 1x1 convolutions and attention. The shim takes those fp16 calls, plus
every strided-batched fp16 call, onto the same kernels with fp16 storage and
fp32 arithmetic; `GFX803_SGEMM_SHIM_NO_FP16=1` passes them back to rocBLAS.

## What is switched off

None of these have ever supported gfx8, on any branch:

- **rocMLIR** -- `AmdArchDb.cpp` knows gfx9/gfx10/gfx11/gfx12 only, verified
  identical from `rocm-5.7.0` through `rocm-6.4.0`.
- **composable_kernel** -- floor is gfx908 across the 6.x line (gfx900 on
  `develop`).
- **hipBLASLt** -- gfx90a and newer.

Turning them off in MIGraphX's cmake is necessary but not sufficient: `rbuild`
builds everything `requirements.txt` lists regardless of MIGraphX's own
options, so `composable_kernel` and `rocMLIR` are stripped from that file
before the build runs. Without that they'd still be compiled for gfx803, and
fail.

hipBLASLt can't be excluded at PyTorch build time -- torch 2.8 links it
unconditionally in `cmake/Dependencies.cmake` -- so the final image sets
`TORCH_BLAS_PREFER_HIPBLASLT=0` to keep GEMMs off a library with no gfx803
kernels.

## Expectations

Untuned MIOpen convolutions, no kernel fusion (no CK, no MLIR), on hardware
with no packed-fp16 and no dot-product instructions. This exists to make a
20-euro card usable, not competitive. For small models it may still beat CPU;
for anything large, measure before believing.

## Versions

| | Pinned to | Why |
| --- | --- | --- |
| Base | `rocm/dev-ubuntu-24.04:6.4.4-complete` | Last ROCm 6 release. `-complete` because MIGraphX needs MIOpen's dev files. |
| Python | 3.12 | Ubuntu 24.04's native interpreter. |
| PyTorch | `release/2.8` + torchvision `v0.23.0` + torchaudio `v2.8.0` | Matches AMD's own `rocm/onnxruntime:rocm6.4.4_ub24.04_ort1.21_torch2.8.0`. |
| ONNX Runtime | `v1.22.2` | Still has the ROCm execution provider's `--use_rocm`/`--rocm_home` build flags, which later releases removed. |

> **PyTorch 2.8 vs 2.6.** The [gfx803_rocm](https://github.com/robertrosenbusch/gfx803_rocm)
> project validated torch `release/2.6` + torchvision `v0.21.0` + torchaudio
> `v2.6.0` on Polaris. This line ships 2.8, validated here for fp32 work (see
> [Verifying on hardware](#verifying-on-hardware)). The 2.6 triple remains a
> fallback through the `PYTORCH_REF`, `TORCHVISION_REF` and `TORCHAUDIO_REF`
> build args.

## Building

The easy way is `./setup.sh`: it checks the machine (Docker and its build
plugin, the GPU devices, a gfx803 card, kernel, RAM, disk space), asks which
parts to build, builds them in order with a log per image in `setup-logs/`,
and confirms that a container can see the GPU. `./setup.sh --check` only runs
the checks; `./setup.sh --all --yes` builds everything without questions.

Underneath, everything builds locally with plain `docker build`, from this
directory, and the commands below do the same by hand. No prebuilt images are
published for this line. (Upstream's GHCR packages, such as
`rocm-migraphx-ort-torch-builder:latest-gfx803`, come from its own pipeline and
are not built from this fork.)

```
docker build -t rocm-gfx803:rocm6.4.4 -f Dockerfile .
docker build -t rocm-gfx803-comfyui:latest -f Dockerfile-ComfyUI .
docker build -t rocm-gfx803-ollama:latest -f Dockerfile-Ollama .
```

`Dockerfile` is one multi-stage build: rocBLAS (with the gfx803 Tensile logic,
several hours), then MIOpen (about 50 minutes), MIGraphX, PyTorch with
torchvision and torchaudio, and ONNX Runtime, each from source and each patched
from `patches/` before it compiles. Docker's layer cache keeps a stage until its
inputs change, so the hours are a one-time cost: editing a MIOpen patch rebuilds
MIOpen and what is built on it (MIGraphX, ONNX Runtime), and the SGEMM shim and
the Python workarounds rebuild only their own small layers. Each patch driver
greps for a marker after applying and stops the build if the patch did not take.

`BUILD_PARALLEL_LEVEL` caps the compile jobs (default: all cores). The
`*_IMAGE` build args take a previously built image for a stage instead of
building it.

## Running

Needs `/dev/kfd` and `/dev/dri` passed through, as any ROCm container does, and
the groups that own them:

```
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    rocm-gfx803:rocm6.4.4 \
    python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The image already sets the Polaris runtime environment
(`HSA_OVERRIDE_GFX_VERSION=8.0.3`, `ROC_ENABLE_PRE_VEGA=1`,
`TORCH_BLAS_PREFER_HIPBLASLT=0`, the SGEMM shim in `LD_PRELOAD`); no extra `-e`
flags are needed. `container-args.sh` holds the device, group and time-zone
arguments the start scripts use, for reuse in your own `docker run` lines.

## Verifying on hardware

The build-time import check in the final stage of `Dockerfile` only
proves the wheels import -- not that MIGraphX, rocBLAS, and MIOpen actually
dispatch to the card correctly. `verify.py` in this repo runs a real
MIGraphX EP inference, a rocBLAS GEMM, and a MIOpen convolution, each checked
against expected output:

```
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v "$(pwd)/verify.py:/verify.py" \
    rocm-gfx803:rocm6.4.4 \
    python3 /verify.py
```

Upstream confirmed all four checks (provider list, MIGraphX EP inference,
rocBLAS GEMM, MIOpen convolution) on a Sapphire RX 470 8GB Mining UEFI. For the
PyTorch side, `training/gfx803_unet_check.py` (in the ComfyUI image at
`/opt/gfx803-training/`) compares fp32 attention, linear, convolution and
normalization, forward and backward, and the whole SD 1.5 UNet against a
float64 CPU reference; it passes on an RX 590 (see
[ComfyUI and SD 1.5 LoRA training](#comfyui-and-sd-15-lora-training)).

If it hangs or the process dies without a Python traceback, check `dmesg` on
the host before suspecting the image -- see the kernel-version caveat below.

## ComfyUI and SD 1.5 LoRA training

`Dockerfile-ComfyUI` builds ComfyUI, a few custom nodes and
[sd-scripts](https://github.com/kohya-ss/sd-scripts) on top of the base image,
each pinned to the commit it was verified with. Build both, from this
directory:

```
docker build -t rocm-gfx803:rocm6.4.4 -f Dockerfile .
docker build -t rocm-gfx803-comfyui:latest -f Dockerfile-ComfyUI .
```

### The gfx803 app

`./gfx803` is the everyday way to use the image: one terminal UI for LoRA
training and the ComfyUI server.

```
./gfx803              # start the service if needed and open the UI
./gfx803 --headless   # from a text console: stop the desktop first, for more VRAM
GFX803_PIN_MCLK=1 ./gfx803 --headless   # also pin the memory clock (stops multi-monitor flicker)
./gfx803 --stop       # shut the service down
./gfx803 --logs       # follow the service's log
```

It starts a long-lived container named `gfx803` whose main process,
`gfx803-service`, runs the training queue and ComfyUI; the UI (`gfx803-app`)
attaches to it with `docker exec`. Quitting the UI stops nothing: training and
ComfyUI keep going, and `./gfx803` from any terminal, a text console or SSH
included, reattaches. Its tabs:

- **Overview**: live graphs of GPU load, power, temperature and VRAM, CPU and
  RAM, plus the current training run (progress, loss, ETA) and ComfyUI's state
  and address.
- **Training**: the queue (add runs through a form, reorder, stop, skip, pause,
  retry, a cool-down between runs) and the run history, with each run's log and
  a button that copies a chosen checkpoint into ComfyUI's LoRA folders: any
  subfolder of `models/loras/` (`SD15/` by default) or a new one, under a short
  editable name such as `MyCharacter-step900.safetensors`, asking before it
  replaces an existing file.
- **Datasets**: each dataset's image and caption counts, trigger word, images
  with transparency, and how many images carry each tag. Tools: WD14 tagging
  on the CPU (only uncaptioned images, all, or adding to existing captions,
  with the trigger put first), setting a trigger word in every caption,
  removing tags (prefilled with those every image has, which the trigger
  should carry instead), and flattening transparent images onto white, since
  training's RGB conversion leaves transparent areas whatever color their
  pixels hold. Flattening also drops the image's cached latents: sd-scripts
  names them by resolution only, so it would otherwise keep training on the
  old pixels. Every change is backed up first under
  `workspace/dataset_backups/` and can be undone step by step. Browse images
  (or Enter on a dataset, or on a tag to see only the images that have it)
  shows each image next to its caption for per-image editing and deleting
  (an image goes into the backups with its caption and cached latents), with a filter
  (`tag`, `-tag`, `uncaptioned`), and a full-screen view with zoom and pan.
  Images are drawn with Sixel or the Kitty graphics protocol where the terminal
  supports one (Konsole, kitty, WezTerm, foot, iTerm2), and as coloured
  half-blocks otherwise; the Linux text console shows only the coarse
  half-block version, and no mouse, so image work there is best done over SSH
  from a graphical terminal. `GFX803_IMAGE_MODE=sixel|tgp|halfcell|unicode`
  overrides the detection.
- **ComfyUI**: start and stop the server, its log, and fp32 instead of
  ComfyUI's default fp16 (fp16 is not validated on gfx803).
- **System**: per-core CPU, memory, disk and network rates, and the host's top
  processes.

Queued runs train one after another. A failed run is marked and the queue moves
on; two failures in a row pause it, since that usually means the GPU or driver
is in a bad state. Training and ComfyUI never use the GPU at once: an 8 GB card
cannot hold SD 1.5 training and a loaded ComfyUI model together. Starting
ComfyUI holds new runs and waits for a run in progress to finish; stopping it
lets the queue continue. `./gfx803 --stop` (or a host shutdown) interrupts a
run the way Ctrl+C would, and the next start resumes it from its last
checkpoint. The queue lives in `workspace/queue/`; `gfx803-queue` inside the
container lists and edits it from the command line.

The container uses the host's network and process namespaces, so the System
tab sees the whole host.

ComfyUI is at `http://localhost:8188` and accepts connections from this machine
only. It has no login, and its Manager can install and run custom code, so
opening it to other devices is a choice to make knowingly: set
`GFX803_COMFYUI_LISTEN=0.0.0.0` (in the shell that runs `./gfx803` or
`./start-comfyui.sh`, or in your shell profile) to make it reachable from the
LAN at `http://<this machine>:8188`, on a network you trust. The setting takes
effect the next time the service starts (`./gfx803 --stop`, then `./gfx803`).

### start-comfyui.sh

`start-comfyui.sh` runs the image without the service, with the GPU passed
through and all data kept next to the script, in git-ignored folders: `ComfyUI/` (models, custom nodes, inputs,
outputs), `miopen/` (MIOpen's find database and kernel cache, which saves a long
search and compile in every new container) and `workspace/` (training data, runs
and LoRA output).

```
./start-comfyui.sh          # ComfyUI on http://localhost:8188 (see GFX803_COMFYUI_LISTEN above)
./start-comfyui.sh bash     # a shell in the container instead
```

The container runs as the user who starts it (plus the host groups that own
`/dev/kfd` and `/dev/dri/renderD*`), so outputs, runs and LoRAs are owned by that
user. `RUN_AS_ROOT=1 ./start-comfyui.sh` runs it as root. Folders left over from
running as root need a one-time
`sudo chown -R "$(id -u):$(id -g)" workspace ComfyUI miopen`.

LoRA training uses the `gfx803-train` wrapper (`training/gfx803-train.sh`),
inside the container:

```
gfx803-train                          # interactive menu (on a terminal)
gfx803-train --list-data              # datasets under /workspace/training_data
gfx803-train --data <set>             # train on one
gfx803-train --help                   # every option
```

Run with no options on a terminal, `gfx803-train` opens a menu: start a new run
(dataset, base model, VAE, steps, learning rate, LoRA rank, clip skip, VRAM
limit, all picked with the arrow keys) or resume an unfinished one. It hands its
choices to the same option parser as the command line and prints the equivalent
command, so a run set up once in the menu can be repeated from a script.
`--menu` opens it explicitly; `GFX803_TRAIN_NO_MENU=1` keeps the old
no-argument behaviour.

A dataset is a folder under `workspace/training_data/` with sd-scripts'
`<repeats>_<trigger>/` subfolders of images and `.txt` captions. Images without
captions can be tagged with sd-scripts' WD14 tagger. It runs on ONNX Runtime's
ROCm provider whenever the image offers one, a path not validated on gfx803:

```
cd /opt/sd-scripts && python finetune/tag_images_by_wd14_tagger.py --onnx \
    --repo_id SmilingWolf/wd-swinv2-tagger-v3 --remove_underscore \
    --always_first_tags <trigger> "/workspace/training_data/<set>/<repeats>_<trigger>"
```

`--train-text-encoder` also trains LoRA weights in the text encoder (CLIP), not
only the UNet, at its own lower rate (`--te-lr`, default 5e-5): it helps a
trigger word or style respond to prompts but overfits quickly. It costs about
10% per step and about 320 MiB of VRAM at 384x384, which matters on an 8 GB card
(see below).

The defaults are
the SD 1.5 base checkpoint at `ComfyUI/models/checkpoints/SD15/v1-5-pruned-emaonly.safetensors`
and `vae-ft-mse-840000-ema-pruned` from `ComfyUI/models/vae/`, fp32, AdamW at
1e-4, LoRA dim and alpha 8, 1200 steps with a checkpoint and resume state every
100. Each run is recorded under `workspace/runs/<name>/` and can be resumed
(`--resume-run`); a run whose loss turns NaN stops, drops the NaN checkpoints
and resumes from the last clean state. A healthy SD 1.5 LoRA loss sits around
0.1-0.2. On an RX 590, a step (4 accumulated images of up to 512x512) takes
about 4 seconds.

`gfx803-train` caps PyTorch's VRAM. Uncapped, PyTorch keeps the memory it
frees and grows until the card is full, whatever training needs: an SD 1.5 run
that needs about 6.1 GB took 7.99 of 8 GB. If the card also drives a desktop,
the driver then moves buffers to system memory and back, steps slow from about
4.3 to 60 seconds, and the compositor fails to place its framebuffers ("Failed
to pin framebuffer with error -12"), which can freeze a monitor until a reboot.
By default the cap leaves what the card's other users hold at startup plus
1 GB for the HIP runtime's context and loaded kernels, measured from amdgpu's
own VRAM counter because ROCm's counts only compute allocations. It warns when
a connected display is left little headroom and stops only if the cap would be
under 3 GB. `--vram-reserve MIB`, `--vram-cap MIB|none` and the
`GFX803_VRAM_RESERVE`/`GFX803_VRAM_CAP` variables override it.

An 8 GB card cannot hold 512x512 SD 1.5 training and a typical desktop at
once: training needs about 6.6 GB under PyTorch plus about 1 GB of runtime
overhead, and a desktop holds 300-700 MiB. `./gfx803 --headless`, run from a
text-console login (Ctrl+Alt+F2), stops the display manager before opening the
app; start the desktop again with `sudo systemctl start display-manager`.

Only fp32 training is validated on gfx803. To check the card, the image and
the shim together, `training/gfx803_unet_check.py` compares fp32 attention,
linear, convolution and normalization (forward and backward) and the whole UNet
against a float64 CPU reference:

```
python3 /opt/gfx803-training/gfx803_unet_check.py \
    --ckpt /opt/ComfyUI/models/checkpoints/SD15/v1-5-pruned-emaonly.safetensors
```

`pip check` in the build log reports conflicts around `opencv-python-headless`;
they are harmless. If the card hangs under load, `tools/gpu-hang-capture/` at the
repository root records what it was running.

## Ollama (local LLMs)

`Dockerfile-Ollama` builds [Ollama](https://github.com/ollama/ollama) v0.35.1
(pinned to its release commit) on the base image. Ollama fetches llama.cpp
b11232 at configure time; its HIP backend is compiled for gfx803 only, against
the base image's ROCm, and Ollama's bundled ROCm copies are left out (they
would add 4.6 GB and are never loaded). Build, from this directory:

```
docker build -t rocm-gfx803:rocm6.4.4 -f Dockerfile .
docker build -t rocm-gfx803-ollama:latest -f Dockerfile-Ollama .
```

A rebuild after a patch change recompiles only the HIP backend, about 15
minutes.

### start-ollama.sh

```
./start-ollama.sh                  # start the server (no-op if it is running)
./start-ollama.sh pull qwen3:8b    # any other arguments run the ollama CLI in it
./start-ollama.sh --logs           # follow its log
./start-ollama.sh --stop           # stop it
```

The server is reachable from this machine only, at `http://127.0.0.1:11434`:
the port is published on the host's loopback address. That address serves
Ollama's own API and the OpenAI-compatible one under `/v1`, which is what
SillyTavern or a coding agent such as Qwen Code points at. Models and Ollama's
key live in `ollama/` next to the script (git-ignored; `OLLAMA_DATA` moves
it). `./start-ollama.sh --help` lists the settings.

The KV cache defaults to 8-bit (`OLLAMA_KV_CACHE_TYPE=q8_0`; `f16` for full
precision). It halves the context's VRAM, which on an 8 GB card decides whether
a context fits next to the model at all: Qwen3 8B with a 16k context needs
2.3 GB of f16 cache and spills layers to the CPU, but runs fully on the GPU
(6.3 GB) with q8_0. With the patches below, q8_0 also generates faster than
f16 from about 4k tokens of context on. Ollama's KV cache type is a server
setting; the OpenAI-compatible API cannot set a context length per request, so
a model meant for long contexts needs a Modelfile with `PARAMETER num_ctx`
(examples in `ollama/import/`).

The image runs AMD's stock HSA runtime; `GFX803_RUNTIME=patched
./start-ollama.sh` loads the base image's patched one instead, which crashes
llama-server with GPU page faults on long prompts. All measurements below were
taken on the stock runtime.

### Patches

`patches/ollama/llama-cpp/` holds llama.cpp patches. Ollama applies every
`*.patch` under its `llama/compat/` to the llama.cpp it fetches, in filename
order; `Dockerfile-Ollama` copies them there and stops the build if one of
them did not apply. Each patch's header gives the full reasoning,
measurements and a reproduction. In short:

- **100-gfx803-mmq-tile-table.patch**: prompt processing ran at 15 tok/s
  (Qwen3 8B Q4_K_M). llama.cpp's quantized matrix multiply used the GCN tile
  table, sized for GPUs with `v_dot4` instructions; on gfx803, which has none,
  the wide K-quant tiles spill hundreds of registers to scratch memory. A
  gfx803 table drops those tiles: 15 -> 157 tok/s (pp512).
- **101-gfx803-fattn-tile.patch**: with flash attention, prompt processing
  fell to a quarter of its speed 8k tokens deep. The flash-attention tile
  kernel uses packed half2 arithmetic on every AMD GPU, but gfx803 has no
  packed half2 instructions, and the extra registers made the wide tiles spill.
  gfx803 now runs those tiles on the kernel's fp32 path. Qwen3 8B pp512 at
  0 / 4096 / 8192 tokens deep: 134 / 55 / 34 -> 163 / 143 / 127 tok/s.
- **102-gfx803-fattn-decode.patch**: token generation with a q8_0 cache slowed
  sharply with context. The vec kernel that generates with a quantized cache
  needs 229 registers per thread on gfx803, so one wave runs per SIMD and its
  memory latency goes unhidden; it also reads each K/V head once per query
  head of a grouped-query model. A new gfx803 kernel serves all query heads of
  a K/V head from one read and stays at 66-104 registers. Generation at
  0 / 4096 / 8192 tokens deep: Qwen3 8B 27.9 / 23.7 / 20.9 -> 28.1 / 26.0 /
  24.4 tok/s, Qwen2.5-Coder 7B 30.1 / 27.1 / 24.8 -> 30.3 / 28.8 / 27.2.

`patches/ollama/ollama/tool-parameters-empty-schema.patch` fixes Ollama
itself, on every GPU: a tool that takes no arguments is re-encoded with
`"properties": null`, which llama.cpp's server rejects (HTTP 400), so clients
that declare such tools (Qwen Code does) fail on their first request. The
patch encodes the empty schema as `{"type":"object","properties":{}}`.

The three llama.cpp patches are gfx803-only by construction: each takes
effect only for `GGML_CUDA_CC_GCN4` and is sized from gfx803 register
measurements. The Ollama patch fixes a bug that is not specific to any GPU.
Upstream llama.cpp master (checked 2026-10-03, b11382) still has the vec
kernel 102 replaces on gfx803, unchanged; llama.cpp issue
[#27796](https://github.com/ggml-org/llama.cpp/issues/27796) reports the same
quantized-KV generation slowdown with context on RDNA4 and was closed as
expected behavior.

### Verifying on hardware

llama.cpp's own `test-backend-ops` compares each GPU operation with its CPU
backend. Build llama.cpp b11232 with the patches applied, `-DGGML_HIP=ON
-DAMDGPU_TARGETS=gfx803`, then:

```
test-backend-ops test -b ROCm0 -o MUL_MAT
test-backend-ops test -b ROCm0 -o MUL_MAT_ID
test-backend-ops test -b ROCm0 -o FLASH_ATTN_EXT -p type_K=q8_0   # also f16, q4_0
```

On an RX 590 all of these pass: MUL_MAT 1301/1301, MUL_MAT_ID 931/931, and
FLASH_ATTN_EXT for f16, q8_0 and q4_0 K/V, including extra grouped-query
generation cases (GQA ratios 2-16, head sizes 64/128/256, with sinks).
Through Ollama, greedy generation from a 5.2k-token prompt matches a CPU run
up to the first near-tie between two tokens, with logprobs within 0.05, for
Qwen3 8B and Qwen2.5-Coder 7B.

### Known issues

- Every model load logs about 4000 `amdgpu: init_user_pages: Failed to get
  user pages: -1` kernel messages. The model loads and runs correctly; the
  cause is not yet investigated. `llama-bench` on the same image does not
  trigger them.
- Ollama is tested on the stock HSA runtime, the image's default. On the
  patched runtime (`GFX803_RUNTIME=patched`) llama-server crashed with GPU page
  faults during a 17k-token prompt, and that runtime reports about 860 MiB less
  free VRAM. A long serving run under the hang watcher has not been done yet.
- With Qwen Code, Qwen2.5-Coder 7B answers without calling tools, and Qwen3 8B
  calls them but misuses Qwen Code's tool-search bridge: it passes
  `read_file`'s arguments straight to `tool_call` without naming the tool.

## Fixed bug case study: ROCMExecutionProvider fused-conv crash (musicnn-class models)

**Fixed** -- both root causes are patched (see [`KERNEL_BUGS.md`](KERNEL_BUGS.md)
for the full record); no workaround is needed anymore. The investigation
below is kept because it's the reference example for how to chase this class
of bug in this repo (isolate, rule out empirically, don't assume).

`ROCMExecutionProvider` (the plain kernel-based HIP EP, distinct from
MIGraphX) is present in this image's ONNX Runtime build as a fallback for
graphs MIGraphX can't compile -- see the [AudioMuse-AI ROCm
plugin](https://github.com/Schaka/audiomuse-rocm-plugin)'s
`plugin/rocm_accelerator/README.md` for the CLAP `keep_aspect_ratio_policy`
parse failure that motivated it. On real gfx803 hardware (Sapphire RX 470
8GB), that fallback **intermittently SIGSEGVs on any graph where ORT's
optimizer fuses Conv+Bias+Activation into a FusedConv node** -- constantly on
CNN-style models (musicnn's embedding/prediction heads), and rarely but
reproducibly on CLAP, whose conv stem is small next to its transformer body
but still enough to reach the fused path.

Root cause, isolated via `MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=6`: ORT's
graph optimizer fuses Conv+Bias+Activation into one node, which ONNX
Runtime's ROCM EP hands to MIOpen's Fusion Plan API rather than three
separate conv/bias/activation calls. On gfx803 that Fusion Plan path --
whether it lands on a fused Winograd solver (`ConvBinWinogradRxSFused`) or
falls through to "No supported fusion solvers found" and a plain-conv
fallback -- corrupts GPU state and eventually faults
(`Memory access fault ... Page not present or supervisor privilege`).

What was ruled out, each confirmed on real hardware, not assumed:

- **Not a stale/mismatched find-db.** `/opt/rocm/share/miopen/db/gfx803_*.fdb.txt`
  ships in 36-CU and 64-CU variants; the RX 470 is 32-CU and correctly falls
  through to its own per-run user db (`gfx803_32...ufdb.txt`) instead of a
  mismatched file -- deleting the shipped fdb files entirely made no
  difference to the crash.
- **Not a single bad solver.** `MIOPEN_DEBUG_CONV_WINOGRAD=0` (disables the
  non-fused Winograd solver) didn't help; `MIOPEN_DEBUG_AMD_FUSED_WINOGRAD=0`
  (disables the fused variant specifically) reduced but did not eliminate the
  crash -- it still faults on the "no fusion solver found" fallback path for
  some shapes.
- **Not a simple async race.** `HIP_LAUNCH_BLOCKING=1` (forces synchronous
  kernel launches) did not prevent the crash.
- **Not deterministic.** The exact same command, same audio sample, same
  shape, same run count crashes on one invocation and completes cleanly on
  the next. Session churn amplifies it heavily (creating and destroying
  sessions reshuffles ORT's arena layout), but a first-session crash in a
  fresh process has been observed too.
- **Not unstable VRAM.** Tried non-mining BIOS, 1750 Mhz VRAM clock, added tons
  of extra cooling for stability and saw no difference.
- **Not SDMA and not an async race** (from the later CLAP investigation):
  `HSA_ENABLE_SDMA=0` and `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` both
  still crash, and the serialized `AMD_LOG_LEVEL=4` trace pins the fault on
  the fused kernels themselves (`miopenSp3AsmConvRxSU_CBA`,
  `MIOpenConvUniBatchNormActiv`) reading past the end of the arena chunk
  holding their weights -- the HSA fault address lands exactly on the chunk's
  end boundary, and dmesg shows `read from 'TC'` (a compute kernel, not a
  copy engine). Whether the over-read faults depends on whether the
  neighbouring page happens to be mapped, which is where the non-determinism
  comes from.

**Update: fixed.** This was not one bug but two, both now fixed and both
documented in full in [`KERNEL_BUGS.md`](KERNEL_BUGS.md):

1. A pervasive rocBLAS/Tensile fp32 GEMM miscompute, fixed by the SGEMM
   shim (now [`patches/sgemm-shim-batched/`](patches/sgemm-shim-batched/), see
   [The SGEMM shim](#the-sgemm-shim)) -- confirmed
   to fix CT2's float32 crash outright, but only measurably *delayed* (not
   eliminated) the ConvActivationFusion crash above, which is what confirmed
   it as a second, separate bug.
2. The actual ConvActivationFusion crash: MIOpen's `ConvOclDirectFwd` solver
   (behind the `MIOpenConvUniBatchNormActiv` kernel this section's own
   investigation already correctly pointed at) has a genuine out-of-bounds
   weights-buffer read for grouped/depthwise convolutions specifically --
   confirmed via a minimal, deterministic, standalone repro with nothing but
   MIOpen involved. Fixed by rebuilding MIOpen from source with
   [`patches/miopen/conv-direct-fwd-grouped-oob.sh`](patches/miopen/conv-direct-fwd-grouped-oob.sh),
   which makes MIOpen's own solver-applicability check reject grouped convs
   outright instead of ever reaching the broken kernel. Confirmed clean over
   40 session churns with `ConvActivationFusion` left fully enabled -- no
   workaround needed anymore; the `optimization.disable_specified_optimizers`
   plugin-side workaround and MIGraphX-instead-of-ROCM routing described
   above are no longer necessary; that also fixed the non-determinism this
   section spent a lot of words on: it was never a "sometimes the neighboring
   page happens to be mapped" thing, it was every grouped-conv dispatch
   reading out of bounds, with the neighboring memory's contents (whatever
   happened to be there) determining whether that particular read actually
   crossed an unmapped page or not.

gfx803 is abandoned upstream with no realistic path to a fix landing there,
so we are not filing upstream reports for either bug -- `KERNEL_BUGS.md`
documents the investigation methodology in full so any future bug in this
class can be chased and patched the same way, in this repo.

## Host-side caveats

Neither of these can be fixed inside the image:

- **Linux 6.12 and 6.13 are reported to segfault on gfx803** (per the
  gfx803_rocm project). If the stack crashes on an otherwise correct setup,
  check the host kernel before suspecting the image.
- Mining-edition cards often have no display outputs, which is irrelevant for
  compute, but confirm the board is a genuine 8GB Polaris 10 and not a 4GB
  rebadge.

## Credit

The ROCm-6.x-plus-rebuilt-rocBLAS approach, and the specific knowledge of which
knobs Polaris needs, comes from two community projects:

- [robertrosenbusch/gfx803_rocm](https://github.com/robertrosenbusch/gfx803_rocm)
  -- actively maintained, ROCm 6.4, the source of the torch 2.6 hardware
  validation and the kernel-version warning
- [woodrex83/ROCm-For-RX580](https://github.com/woodrex83/ROCm-For-RX580) --
  the earlier ROCm 6.1.2 take

Neither builds MIGraphX or ONNX Runtime; that part is new here, and has since
been verified on real hardware (a Sapphire RX 470 8GB Mining UEFI) with
[`verify.py`](#verifying-on-hardware).
