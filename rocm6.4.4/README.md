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
> case study](#case-study-rocmexecutionprovider-fused-conv-crash-musicnn-class-models)
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
| PyTorch | `ROCm/pytorch` `release/2.8` | No gfx803 wheel has ever been published. torchvision and torchaudio are built alongside it. `patches/pytorch/` routes int8 matmul to rocBLAS (its default hipBLASLt path has no gfx803 kernels) and fixes four kernels that mix the 64-lane hardware wavefront with PyTorch's 32-lane `C10_WARP_SIZE` on gfx803 (layer_norm, top-k's scan, index_put) or lose their launch to a rocSPARSE symbol clash (rocPRIM: nonzero, masked indexing, unique, large sorts). Each header gives the cause and the check. |
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
results and run large GEMMs at 3.9-4.8 TFLOPS on an RX 590, 3-4x stock
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
with no packed-fp16 and no dot-product instructions. This exists to make an
inexpensive, unsupported card usable, not competitive. For small models it may still beat CPU;
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

Optional: `../tools/ComfyUI-gfx803-attention/install.sh` adds a custom node
with gfx803 kernels for attention, GroupNorm and GEGLU, which makes an SD 1.5
fp32 step about 19% faster (697 -> 567 ms at 512 px) and avoids the 2.3 GB
score matrix PyTorch's attention allocates. It compiles the kernels with the
base image's `hipcc` when the host has none and copies the node into
`ComfyUI/custom_nodes/`; restart ComfyUI afterwards. The node acts only on
inference (calls that need gradients go to PyTorch), so training is
unaffected. Remove the `ComfyUI/custom_nodes/ComfyUI-gfx803-attention/`
folder to turn it off. Details: `docs/FINDINGS.md` section 6.

### start-comfyui.sh

`start-comfyui.sh` runs the image with the GPU passed through and all data
kept next to the script, in git-ignored folders: `ComfyUI/` (models, custom
nodes, inputs, outputs), `miopen/` (MIOpen's find database and kernel cache, which saves a long
search and compile in every new container) and `workspace/` (training data, runs
and LoRA output).

```
./start-comfyui.sh          # ComfyUI on http://localhost:8188 (see below for GFX803_COMFYUI_LISTEN)
./start-comfyui.sh bash     # a shell in the container instead
```

The container runs as the user who starts it (plus the host groups that own
`/dev/kfd` and `/dev/dri/renderD*`), so outputs, runs and LoRAs are owned by that
user. `RUN_AS_ROOT=1 ./start-comfyui.sh` runs it as root. Folders left over from
running as root need a one-time
`sudo chown -R "$(id -u):$(id -g)" workspace ComfyUI miopen`.

ComfyUI accepts connections from this machine only. It has no login, and its
Manager can install and run custom code, so opening it to other devices is a
choice to make knowingly: set `GFX803_COMFYUI_LISTEN=0.0.0.0` to make it
reachable from the LAN at `http://<this machine>:8188`, on a network you
trust.

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
`--menu` opens it explicitly; `GFX803_TRAIN_NO_MENU=1` skips the menu and starts a run with the defaults.

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
overhead, and a desktop holds 300-700 MiB. Train from a
text-console login (Ctrl+Alt+F2) after stopping the display manager
(`sudo systemctl stop display-manager`); start the desktop again with
`sudo systemctl start display-manager`.

`gfx803-train` trains in fp32, the most thoroughly checked precision on gfx803.
fp16 training works too (section 8 of `docs/FINDINGS.md` has the measurements:
1.7 GB less VRAM at similar speed) but is run with sd-scripts directly. To check the card, the image and
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
(see [Fitting models into 8 GB](#fitting-models-into-8-gb)).

A loaded model stays loaded until you unload it (`OLLAMA_KEEP_ALIVE=-1`;
`./start-ollama.sh stop MODEL` frees it). Ollama's own
default of 5 minutes unloads it between agent sessions, and reloading the large
models takes tens of seconds to minutes. Set `OLLAMA_KEEP_ALIVE=30m` or similar
to get a timeout back.

**VRAM margin.** llama.cpp fits each model to the VRAM that ROCm reports free
and keeps a margin, 1 GiB by default. On this card ROCm's figure leaves out the
display's framebuffers and other programs' buffers, so with a desktop running
most of that margin is already gone, and a model plus its working buffers can
fill the card. When that happens the compositor cannot pin its framebuffers
("Failed to pin framebuffer with error -12"), windows stop updating and
applications lose their GPU contexts. `start-ollama.sh` therefore reads the
card's in-use VRAM from amdgpu (which does count the desktop) when it starts
the server, subtracts what other ROCm processes hold (a TTS server or ComfyUI;
ROCm's free figure already leaves that out), and sets the margin to the rest
plus `GFX803_OLLAMA_HEADROOM_MIB` (default 512). Start the server with your
usual programs open, and after any other GPU service such as a TTS server;
with the display manager stopped, the margin shrinks to about the headroom
alone and leaves the most room for context. `LLAMA_ARG_FIT_TARGET` (MiB) sets
the margin outright. Measured with Chatterbox TTS (2.9 GB) already running:
counting its memory in the margin left Qwen3.6 35B-A3B 1.6 GB of the card and
15 tok/s; subtracting it gives 4.4 GB and 24.6 tok/s, with the card at 7.5 of
8 GB and speech requests still working.

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
- **103-gfx803-mmvq-two-rows.patch**: token generation reads every weight once
  per token, but the K-quant and low-bit IQ matmuls reached only 127-165 GB/s
  of the card's ~225 (q8_0 and f16 reach it): without integer dot product
  instructions their dot products are long ALU sequences, so the kernel is
  instruction-bound. For one token, one warp over two rows per block halves
  the per-row reduction and launch overhead: q4_K +11-27%, q3_K +31%, q6_K +9%.
  Types that were already at the bandwidth limit, or measured slower that way
  (q5_K, q8_0, iq4_xs, q5_1), keep the GCN shape. Generation at 0 / 8192
  tokens deep: Qwen3 8B 27.9 / 24.2 -> 31.4 / 26.5 tok/s, Qwen2.5-Coder 7B
  30.3 / 27.0 -> 34.6 / 30.1.
- **104-gfx803-sdwa-dp4a.patch**: in the decode kernels, the 4-byte integer
  dot product under every quantized matmul is built from SDWA byte-select
  multiplies instead of unpacked bytes, which frees registers: decode for 2-8
  tokens no longer spills or drops to one wave, and 4-8 tokens get up to 1.6x
  faster. Prompt kernels and the IQ4/FP4 table-lookup types keep the
  compiler's code, which measured faster there. With a 5.2k-token
  context, Qwen3 8B and Qwen2.5-Coder 7B (Q4_K_M) generate 6-7% faster; prompt
  speed and IQ4_XS models are unchanged. Output is bit-identical.
- **105-gfx803-q5k-decode.patch**: q5_K's decode dot product reads its scales
  with one 16-byte load instead of four (q5_K matmul about 4% faster with
  104).
- **106-gfx803-cublas-prompt.patch**: prompt batches of quantized matmuls ran
  through the int8-emulating mmq path at about a third of the card's fp32 rate.
  They now dequantize the weights into padded fp32 rows and multiply on the
  SGEMM shim (padding avoids the DRAM-channel conflict that 8 KiB row strides
  cause), with a cap on the converted weights' memory. Qwen3 8B pp512
  163 -> 223 tok/s; a 5.2k-token prompt through Ollama 149 -> 215 tok/s.
  Results are closer to the CPU backend's (KL divergence 0.0103 -> 0.0086).
- **107-concat-transposed-src1.patch**: the concat that feeds Gated DeltaNet
  layers (Qwen3.5/3.6, Qwen3-Next) read its transposed operand with a 32 KiB
  stride, 27 ms per call. A tiled kernel takes it to 1.14 ms.
- **108-moe-expert-prefetch.patch**: when a MoE model keeps most experts in
  system RAM, each layer's experts were copied to the card while the GPU
  waited. Large batches now prefetch the next layer's experts on a second
  stream, over the DMA engines (`HSA_ENABLE_SDMA=1`, set in
  `Dockerfile-Ollama`). Qwen3.6 35B-A3B pp2048 at 512 / 1024 / 2048-token
  batches: 193 / 250 / 295 -> 243 / 321 / 367 tok/s.
- **109-sched-batched-host-copies.patch**: during generation a MoE layer's CPU
  expert split waited for each of its GPU inputs separately, about 130 GPU round
  trips per token. The copies are now queued and waited for once. Qwen3.6
  35B-A3B tg128 23.8 -> 24.5 tok/s.

`patches/ollama/llama-cpp-server/090-mmproj-fit-target-device.patch` fixes
llama.cpp's server, on every GPU. When it fits a model with a multimodal
projector, it reserves the projector's memory under the wrong device index
(the backend registry's, which also holds the CPU, instead of the model's
device list). With the projector on the GPU nothing is reserved for it, so the
model fills the GPU and encoding an image then needs memory that is not
there; with the projector on the CPU, about 1.1 GB of VRAM is held back for
nothing. llama-server is built in Ollama's CPU stage, so `Dockerfile-Ollama`
copies this folder in before that stage and the HIP patches after it.

`patches/ollama/ollama/` holds patches to Ollama itself, both for bugs on every
GPU:

- **tool-parameters-empty-schema.patch**: a tool that takes no arguments is
  re-encoded with `"properties": null`, which llama.cpp's server rejects (HTTP
  400), so clients that declare such tools (Qwen Code does) fail on their first
  request. The patch encodes the empty schema as
  `{"type":"object","properties":{}}`.
- **gpu-refresh-timeout.patch**: before loading a model, Ollama refreshes each
  GPU's free memory by starting a small llama-server, with a 3-second deadline,
  and keeps the previous values when it misses. The ROCm library alone takes
  about 3 s to start on gfx803 and a refresh about 8 s, so every refresh missed
  and Ollama kept counting the memory of models it had already unloaded. Its
  decisions then went wrong, for example encoding images on the CPU because a
  vision projector seemed not to fit on an empty card. The refresh deadline is
  now twice the first discovery's duration, between 3 and 30 s; GPUs that start
  quickly keep 3 s.

The llama.cpp kernel patches 100-104 and 106 are gfx803-only by construction:
each takes effect only for `GGML_CUDA_CC_GCN4` or `__gfx803__` and is sized from
gfx803 register and bandwidth measurements. 105 changes q5_K's dot product for
every GPU, with the same arithmetic. 107, 108, 109, 090 and the two Ollama
patches fix problems that are not specific to any GPU (107-109 were measured
on gfx803 only).
Upstream llama.cpp master (checked 2026-10-03, b11382) still has the vec
kernel 102 replaces on gfx803, unchanged; llama.cpp issue
[#27796](https://github.com/ggml-org/llama.cpp/issues/27796) reports the same
quantized-KV generation slowdown with context on RDNA4 and was closed as
expected behavior.

### Fitting models into 8 GB

A model's context length is part of its Ollama entry. To make a long-context
entry from a model you already have, write a Modelfile and create it inside the
container (models live in `ollama/`, mounted at `/home/gfx803/.ollama`):

```
# ollama/import/Modelfile-mymodel-32k
FROM qwen3:8b
PARAMETER num_ctx 32768
```
```
./start-ollama.sh create mymodel:32k -f /home/gfx803/.ollama/import/Modelfile-mymodel-32k
```

A new entry shares the original's weight files, so it costs no disk space.
Load it once and check `./start-ollama.sh ps`: the processor column should
read 100% GPU. Clients such as SillyTavern send their own context size with
each request, which replaces the entry's; set the client to the same value.

What decides how much context fits, measured on an RX 590 with a desktop
running:

- **Architecture.** In hybrid models such as Qwen3.5, only every fourth layer
  keeps a growing KV cache, so context costs about 22 KB per token (q8_0
  cache, working buffers included) against several times that for Qwen3 8B.
  Qwen3.5 9B IQ4_XS runs a 64k context fully on the GPU, generating at 31
  tok/s, and reads prompts at 156 tok/s empty and 120 tok/s 30k tokens deep.
- **Vision projectors.** A vision model's projector (0.9 GB for Qwen3.5) sits
  on the GPU, and its worst case while encoding an image (about 1.1 GB) is
  reserved too. The same model at 16k with the projector reads a 512x512 image
  in 3.5 s and a 1080p one in 56 s. An entry made from the model's weights
  alone, without the projector, gets that memory back for context; keep a
  vision entry for images and a text entry for long conversations.
- **Full-precision output tensors.** Some uploads keep `output.weight` in bf16,
  and with a large vocabulary it can be a third of the model (2 GB of 6.4 for
  that Qwen3.5 9B). Every generated token reads it.
  [`ollama-tools/requant-output-head.py`](ollama-tools/requant-output-head.py)
  converts just that tensor to q8_0 and copies the rest unchanged: 0.95 GB
  less VRAM, generation +11%, the same top token as the original 98% of the
  time. Import the result with a Modelfile whose `FROM` is the new file.
- **Mixture-of-experts models.** A MoE far larger than the card still runs:
  llama.cpp keeps attention, shared weights and the KV cache on the GPU and
  the routed experts in system RAM. `ollama/import/Modelfile-qwen3.6-35b-256k`
  runs Qwen3.6 35B-A3B at its full 262k context (189 tok/s prompt, 22 tok/s
  generation with DDR4-2800, about 20 GB of RAM). Its comments explain the
  settings: the MTP draft head off, 1024-token batches and `use_mmap false`.
  Such a model's speed follows system RAM speed and CPU cores.
  `Modelfile-qwen3-coder-next-heretic-256k` does the same for an 80B-A3B
  coding model imported from a GGUF (105 tok/s prompt, 18 tok/s generation,
  about 50 GB of RAM); a raw GGUF needs the `RENDERER`/`PARSER` lines of
  Ollama's own build of the model, or clients' tool calls are not recognised.

**Thinking.** Ollama turns reasoning on for models that support it unless the
client sends `"think": false`. SillyTavern's Text Completion mode builds the
prompt itself (Ollama's raw mode), so there the switch is the reply prefix:
**Start Reply With** set to `<think>`, a blank line and `</think>` makes Qwen3
and Qwen3.5 answer directly. SillyTavern's Ollama image captioning sends no
`think` field, so captions from such a model begin with its reasoning.

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
  Qwen3.6 35B-A3B (see Fitting models) reads, edits and runs files correctly.
- `test-backend-ops` passes everything except `SOLVE_TRI` above 64x64
  (7 cases); no model used here reaches that path.
- Images are slow to read: a 1080p image becomes about 2000 tokens and takes
  56 s with Qwen3.5 9B, where 2000 text tokens take about 15 s. Where the time
  goes in the vision path is not yet measured.
- Switching models takes about 8 s longer than Ollama intends, while it
  refreshes free memory (see gpu-refresh-timeout.patch above); the alternative
  was decisions made from stale figures.

## Case study: ROCMExecutionProvider fused-conv crash (musicnn-class models)

`ROCMExecutionProvider` (the plain kernel-based HIP execution provider, distinct
from MIGraphX) is in this image's ONNX Runtime build as a fallback for graphs
MIGraphX cannot compile; the [AudioMuse-AI ROCm
plugin](https://github.com/Schaka/audiomuse-rocm-plugin) uses it for CLAP's
`keep_aspect_ratio_policy` parse failure. On gfx803 it intermittently crashed
with a memory access fault on any graph where ONNX Runtime fuses
Conv+Bias+Activation into a `FusedConv` node: constantly on CNN-style models
such as musicnn, rarely on CLAP, whose small conv stem still reaches the fused
path. Two separate bugs caused it, both fixed and documented in full in
[`KERNEL_BUGS.md`](KERNEL_BUGS.md):

1. rocBLAS/Tensile's fp32 GEMMs return wrong results. The
   [SGEMM shim](#the-sgemm-shim) fixes that, which removed CT2's float32 crash
   but only delayed the fused-conv one.
2. MIOpen's `ConvOclDirectFwd` solver reads past the end of its weights buffer
   for grouped and depthwise convolutions. The fused kernels
   (`MIOpenConvUniBatchNormActiv`, `miopenSp3AsmConvRxSU_CBA`) hit it, and
   whether the read faults depends on whether the neighbouring page is mapped,
   which is why the crash looked random. A serialized `AMD_LOG_LEVEL=4` trace
   puts the fault address exactly on the end of the arena chunk that holds the
   weights, and `dmesg` shows a read from `TC` (a compute kernel, not a copy
   engine).
   [`patches/miopen/conv-direct-fwd-grouped-oob.sh`](patches/miopen/conv-direct-fwd-grouped-oob.sh)
   makes the solver's applicability check reject grouped convolutions. With
   `ConvActivationFusion` left enabled, 40 session churns ran clean.

A stale find-db, a single bad solver (`MIOPEN_DEBUG_CONV_WINOGRAD=0`,
`MIOPEN_DEBUG_AMD_FUSED_WINOGRAD=0`), unstable VRAM clocks, SDMA and an
asynchronous race (`HIP_LAUNCH_BLOCKING=1`, `HSA_ENABLE_SDMA=0` and serialized
launches all still crashed) were each ruled out. To chase a similar bug, isolate
the layer with a standalone MIOpen reproduction, serialize launches, and read
the fault address against the allocation map; `KERNEL_BUGS.md` describes the
method.

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
