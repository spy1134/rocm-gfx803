# rocm-gfx803: modern AI tools on AMD Polaris cards

## Quick Start

Requires Linux, Docker, and a Polaris (gfx803) GPU such as the RX 470-590.

```
git clone https://github.com/spy1134/rocm-gfx803.git
cd rocm-gfx803/rocm6.4.4
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

## How fast is it?

Measured on an RX 590 8 GB. For images and training, "before" is an early setup that already
computed correct results (the original stack was faster in places, but its results were wrong);
for Ollama, it is the same llama.cpp version without this project's changes.

| Task | Before | Now | |
| --- | --- | --- | --- |
| Generate an SD 1.5 image (20 steps at 512 px) | 35 s | 10.5 s | 3.3x |
| One SD 1.5 LoRA training step (4 images, 512 px) | 12.1 s | 3.4 s | 3.6x |
| Ollama reads a prompt (Qwen3 8B) | 14 tokens/s | 223 tokens/s | 16x |
| A coding agent's first request (a 12,000-token prompt) | 18 min | 71 s | 15x |
| Ollama writes text (Qwen3 8B) | 27 tokens/s | 30.5 tokens/s | 1.1x |

Image generation is ComfyUI in fp32 with the optional gfx803 node (12.9 s without it). Writing
text is limited mostly by how fast the card can read the model's weights, so it had the least
room to grow; its gain increases with the length of the conversation (20.9 to 26.5 tokens/s with
8,000 tokens of context). Some changes don't show up as speed at all: half-precision training
uses 1.7 GB less VRAM, which makes SDXL LoRA training fit on 8 GB, and an 8-bit KV cache keeps a
16k-context Qwen3 8B entirely on the card. [`docs/FINDINGS.md`](docs/FINDINGS.md) has the
measurements behind each number, and [`docs/OPTIMIZATION_TARGETS.md`](docs/OPTIMIZATION_TARGETS.md)
what could still get faster.

## What was broken, and is fixed here

In short, out of the box on these cards: training learned from wrong numbers, sustained loads
could freeze the GPU or the whole machine, core PyTorch functions computed wrong results,
failed to run, or hung the card (layer_norm, top-k sampling, boolean masks, sorting), and
Ollama read prompts about ten times slower than it should, generated text well below what the
card's memory allows, and could fill the card until the desktop stalled. Each of these is fixed,
and the fixes made fp16 and local language models run at full speed. [`docs/FINDINGS.md`](docs/FINDINGS.md) has
the details of every problem, its cause, the fix, and how to check it yourself.

## Known issues

- The patched GPU runtime library from upstream page-faults on this card in some workloads (it
  crashed Ollama mid-prompt), so the images use AMD's stock one; `GFX803_RUNTIME=patched` selects
  the patched one for comparison. Why it faults is not yet known. ComfyUI's occasional GPU fault
  on the second image after switching models may be the same bug.
- Mining-edition cards often ship with memory clocked above its rating, which causes random GPU
  faults and freezes. If you see those, check your card's memory clock first (see "Host VBIOS
  setting" in [`README-rocm10.md`](README-rocm10.md)).
- Linux kernels 6.12 and 6.13 have been reported to crash with these cards.
- When a GPU program hits a failed safety check (a device-side assert), these cards do not stop
  the program with an error: the GPU hangs and the driver resets it, which wipes VRAM for every
  program using it. A desktop on the card usually survives the reset; GPU programs need a
  restart. The PyTorch bugs that triggered this are fixed, but any new one will look the same.

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
