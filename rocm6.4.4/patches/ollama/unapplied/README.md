# Patches that are kept but not built

`109-sched-batched-host-copies.patch` batches the device-to-host copies of the inputs of CPU-side splits in
llama.cpp's graph scheduler. It is not applied by `Dockerfile-Ollama`.

Why: the scheduler lives in `libggml-base`, so a scheduler patch only takes effect when it is applied before
the CPU build (see `llama-cpp-sched/`). With this patch in the scheduler and `HSA_ENABLE_SDMA=1` (the image
default), decoding a MoE model with the experts in host memory ends in `amdgpu: ring sdma0 timeout` and a
BACO reset of the GPU. The same decode passes on the scheduler without the patch, and with the patch
when SDMA is off. It also gave no measurable decode speedup (Qwen3.6 35B-A3B, 28 to 31 tokens/s with and
without it). The cause of the SDMA timeout is not understood.
