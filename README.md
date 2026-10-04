# rocm-gfx803: modern AI tools on AMD Polaris cards

## Quick Start

Requires Linux, Docker, and a Polaris (gfx803) GPU such as the RX 470-590.

```
cd rocm6.4.4
./setup.sh                        # verify the system and build the images (several hours on first run)
./gfx803                          # launch the app: LoRA training, ComfyUI, dataset tools, GPU monitor
./start-ollama.sh run qwen3:8b    # chat with a local language model
```

ComfyUI is served at http://localhost:8188 once started from the app. Full documentation follows.

---

Got an RX 470, 480, 570, 580 or 590 (or another "gfx803" Polaris card) and want to use it for
AI? AMD dropped these cards from ROCm, its GPU computing platform, after version 6.0, and most
guides you find are either out of date or quietly produce wrong results. This project packages a
working, tested setup:

- **Make images** with Stable Diffusion 1.5 in [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- **Train your own LoRAs** (small add-on models for a character or style) for SD 1.5, through a
  terminal app with a training queue, live GPU graphs and a dataset manager.
- **Chat with local language models** through [Ollama](https://ollama.com), for SillyTavern,
  coding assistants and the like.

Everything runs in Docker on Linux. Everything is tested on a real card, an RX 590 8 GB, against
reference results computed on the CPU, because on this hardware the usual problem is not a crash
but an answer that is silently wrong.

## What you need

- A Polaris card. 8 GB of VRAM is strongly recommended; training needs it.
- A Linux machine with Docker.
- About 100 GB of free disk space, and a few hours for the first setup: the main libraries are
  compiled for your card from source. The finished setup takes about 31 GB, and later updates
  reuse most of the work.

## Getting started

1. **Install Docker** from your distribution (for example `sudo pacman -S docker docker-buildx`
   on Arch, or `sudo apt install docker.io docker-buildx` on Ubuntu), then start it and allow
   your user to use it:
   ```
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER
   ```
   Log out and back in afterwards.
2. **Download this project** (the green "Code" button on GitHub, or `git clone`), open a
   terminal in its `rocm6.4.4` folder, and run:
   ```
   ./setup.sh
   ```
   It checks your machine and tells you exactly what to fix if anything is missing, asks what
   you want to set up, and builds it. You can stop it with Ctrl+C and run it again later; it
   continues where it stopped.
3. **Start using it:**
   ```
   ./gfx803                       # the app: training, ComfyUI, dataset tools, GPU monitor
   ./start-ollama.sh pull qwen3:8b    # download a language model
   ./start-ollama.sh run qwen3:8b     # chat with it
   ```
   ComfyUI runs in your browser at http://localhost:8188 once you start it from the app.

[`rocm6.4.4/README.md`](rocm6.4.4/README.md) explains each part in detail: the app, training
options, ComfyUI and Ollama.

## How stable is it?

- Every fix was checked by comparing the card's results with the same math done on the CPU,
  operation by operation and for whole models.
- LoRA training runs for hours without hangs or bad results, and both the Stable Diffusion model
  and Ollama's language models produce the same output on the card as on the CPU.
- Half-precision (fp16) works too, and frees enough memory to make training SDXL LoRAs on an
  8 GB card possible (tested in short runs; not yet built into the app).
- Tested on one RX 590. Other Polaris cards use the same chip design and should behave the same,
  but their clocks, memory and firmware differ.

## What was broken, and is fixed here

In short, out of the box on these cards: training learned from wrong numbers, sustained loads
could freeze the GPU or the whole machine, a core PyTorch function computed wrong results, and
Ollama read prompts about ten times slower than it should. Each of these is fixed, and the fixes made
fp16 and local language models run at full speed. [`docs/FINDINGS.md`](docs/FINDINGS.md) has
the details of every problem, its cause, the fix, and how to check it yourself.

## Known issues

- ComfyUI has occasionally hit a GPU fault on the second image after switching models. The cause
  is still being tracked down; the leading suspect is the GPU runtime library the base image
  ships, and the start scripts can use AMD's stock one instead (`GFX803_RUNTIME=stock`).
- Mining-edition cards often ship with memory clocked above its rating, which causes random GPU
  faults and freezes. If you see those, check your card's memory clock first (see "Host VBIOS
  setting" in [`README-rocm10.md`](README-rocm10.md)).
- Linux kernels 6.12 and 6.13 have been reported to crash with these cards.

## Where this comes from

This is a fork of [Schaka/rocm-gfx803](https://github.com/Schaka/rocm-gfx803), which built the
original ROCm 6.4.4 setup and its first set of fixes. Upstream has moved on to ROCm 10 and no
longer maintains this version; this fork does. Only the `rocm6.4.4/` folder and `tools/` are
maintained here. The rest of the repository (the ROCm 10 build at the root and the archived
`rocm7.14/`) is inherited from upstream and left as it was; its documentation is
[`README-rocm10.md`](README-rocm10.md).

## Credit

The original 6.4.4 setup and its first fixes are [Schaka](https://github.com/Schaka)'s (see
[`rocm6.4.4/KERNEL_BUGS.md`](rocm6.4.4/KERNEL_BUGS.md)). The approach of using ROCm 6.x with
libraries rebuilt for gfx803 comes from the community projects
[robertrosenbusch/gfx803_rocm](https://github.com/robertrosenbusch/gfx803_rocm) and
[woodrex83/ROCm-For-RX580](https://github.com/woodrex83/ROCm-For-RX580).
