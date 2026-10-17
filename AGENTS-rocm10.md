# Rules that apply to the ROCm 10 line only

These sections were moved out of `AGENTS.md` so that work on `rocm6.4.4/` does not load them. They are unchanged.
Read this file only when asked to work on the root (ROCm 10) line.

#### Every local build sets TENSOR_TOPK_OPT_LEVEL=-O0.

Export `TENSOR_TOPK_OPT_LEVEL=-O0` before every local `docker buildx bake` that
builds the `pytorch` target. That includes a bake of `final`, `vllm`,
`torchvision` or `torchaudio`, because each of them pulls `pytorch` in.
`docker-bake.hcl` defaults the value to `-O3`. ROCm 10.1's clang 24 does not
finish the gfx803 device code of the one file `TensorTopK.hip` at -O1 or higher.
Its SelectionDAG instruction selection grows by about 20 MB/s with no end, and a
96 GB host ran out of memory on it. The same file at -O1 compiles in minutes for
gfx900 and with ROCm 10.0's clang 23, so the cause is a gfx803-only fault in
the 10.1 compiler, not the build setup. `scripts/build/pytorch.sh` has the
measurements. The level reaches only that one file, through the compiler wrapper
in that script. Before you wait on a build, make sure that its log says
`TensorTopK at -O0`.

## Runner disk budget for the final stage

The `final` bake target has failed on the hosted runner's disk more than once.
Each time, this happened right after a new from-source component was wired
into it: first rocsolver, then triton. The failure looks like a caching bug at
first. `docker buildx bake --push` dies partway through `final-wheels.sh` with
`No space left on device`. `scripts/lib/bake-retry.sh`'s 429 retry loop used to
retry this failure five times anyway. That burned up to forty minutes to fail
the same way each time. `bake_push_with_retry` now recognizes this error and
stops after the first attempt.

Read the actual CI log's byte counts before you assume a cache problem. Run
`gh run view --job=<id> --log` on a `final` job. It shows every layer pull with
its size. Check `df -h / /mnt` right after the "Free disk space" step. It
shows what the runner actually had to work with.

The real cause each time was the same: `final.Dockerfile` pulls more than it
reads. A `COPY --from=<component> ...` instruction needs that component's
entire published image on disk before it can copy even one file out of it. A
component image that carries its whole build environment costs the runner
that component's full size. Examples of such an environment: a compiled ROCm
SDK copy, a git checkout with submodules, a build tree, or (for triton) a
from-source LLVM/MLIR build. This cost applies even when `final` reads only
one small directory from that image.

`docker-bake.hcl`'s `wheels_image()` and `wheels_ctx()` functions fix this for
the five components it hit: pytorch, torchvision, torchaudio, ort, and triton.
Each of these publishes a second, trimmed image. That image comes from a
`FROM scratch AS wheels` stage in the component's own Dockerfile. The stage
copies out only what `final` reads: `/wheels`, or `/onnxruntime/dist` for ort.
Bake tags this image `<image>-wheels`. It builds in the same `docker buildx
bake --push` invocation as the full image, so the shared `builder` stage does
not compile twice. See `scripts/build/...` for where each component writes
its wheel. See the `targets=` case in
`.github/workflows/build-component.yml` for which components get the extra
tag. `final`'s own contexts pull the `-wheels` tag for all five. migraphx,
rocr-clr, miopen, and rocsolver still publish only their normal image. `final`
reads a large fraction of migraphx's `/opt/rocm`. The other three already
publish an image close in size to what `final` copies out of it.

Apply the same rule when you wire a sixth from-source component into `final`
the way triton was wired in. Before you add a `COPY --from=<new-component>` in
final.Dockerfile, check the new component's Dockerfile for a build
environment beyond what `final` actually reads: a git checkout, a compiled
toolchain, an LLVM build, or apt build tools. If you find one, give that
component the same `FROM scratch AS wheels` split and a matching
`wheels_image()`/`wheels_ctx()` pair. Do this before you wire the component
into `final`. Do not wire in the full image first and find out on a hosted
runner that it does not fit.

## Component pins: branches, not commit SHAs, and no nightlies

Every upstream component that this repo builds from source is pinned to a named
release branch. On the 10.x line the pins are variables in `docker-bake.hcl`:
`ROCM_SYSTEMS_REF`, `ROCM_LIBRARIES_REF`, `MIGRAPHX_REF` and `PYTORCH_REF`.
`docker buildx bake --print pins` lists all of them. The three ROCm refs, the
base image, the release tag, the line and the `/opt/rocm/core-<major.minor>`
path all derive from the one `ROCM_VERSION` variable, so a move to another 10.x
release changes that value and nothing else in the build files. The derived
values are `release/therock-<major.minor>`, `release/rocm-rel-<major.minor>`,
and similar. Do not pin a frozen commit SHA, and do not resolve `develop` or
`main` per run. The mainline repo (`rocm-migraphx-ort-builder`) uses the same
convention in its `release.yml`. The archived `rocm7.14/` line uses `release/therock-7.14` and
`release/rocm-rel-7.14`.

This is safe only because CI here runs on manual dispatch and has no schedule.
No nightly job re-runs against a moving tip while nobody watches. A branch pin
means: build what that branch holds on the day a person runs the workflow. Two
manual runs weeks apart can land different commits, because upstream can push a
cherry-pick to the branch in between. That is expected. It is not drift to chase. If
a build starts to fail and nothing in this repo changed, compare the branch tip
with the commit that the last good build used, before you suspect a local
regression.

Two additions to that rule matter for how the build finds the tip. CI resolves
each branch to its commit once per run and passes it as a `*_SHA` build-arg, so
a moved tip changes the layer cache key instead of being reused invisibly. See
"Component images, pins and line provenance" in `README.md`. Nobody sets these
values by hand. `docker-bake.hcl` keeps the branch names as the readable pin,
and `docker buildx bake --print pins` shows them resolved.

If you add a component that has no release branch, pin it to an exact commit
SHA. Say why in a comment next to it. This happens for a component
whose upstream only tags releases, or for rocBLAS and MIOpen before the
`rocm-libraries` monorepo restructure. Do not answer the question by defaulting
to `develop` or `main`.

## vLLM lives in vllm/ as a hard fork, not as patch files and not as a submodule

`vllm/` in this repo is a hard fork. It carries upstream gfx906 vLLM support,
adjusted for gfx803, and this repo's own git history tracks it directly as of
2026-08-29. It began as a separate checkout with its own `.git` and a remote
named `ai-infos/vllm-gfx906-mobydick`. That link was removed on purpose: no
`.git`, no `.gitmodules`, no gitlink, and no independent history. The decision
was discussed and explicit. The fork will never be upstreamed, so a separate
history to reconcile later has no value. Do not run `git submodule add` on it,
and do not reconnect it to that remote unless you are told to.

The fork lives on the 10.x root line. It was copied there when the 7.14 line was
archived under `rocm7.14/` on 2026-08-29. The untouched fork copy stays under
`rocm7.14/vllm/` for the record. The fork targets the 10.x stack by assumption.
Its hand-written gfx803 kernels in `vllm/vllm/gfx803_kernels/*.hip` are
version-agnostic source. They compile once with the stack's own
`hipcc --offload-arch=gfx803`, and each loader's docstring gives the exact
command. `librocblas.so` resolves through the stack's `LD_LIBRARY_PATH`, which
is `/opt/rocm/core-<major.minor>/lib`. The compiled `.so` files are built on the
box next to their loaders and are never committed, so this repo pins nothing
stack-specific.

Hardware validation of vLLM ran on the ROCm 10.0 stack on 2026-09-02, and it is
open on the current 10.x pin. Two crashes blocked it, and both were in the ROCm
stack rather than in vLLM.
`patches/rocm-systems/va-reuse-defer-noremap.patch` fixes the first one. The
va-reuse-defer park branch re-mapped a buffer in `_fmm_map_to_gpu` and left a
kernel GPUVM mapping behind. libhsakmt's aperture allocator handed that range
out again, so every code-object load collided with it. The kernel load returned
EINVAL, that became `HSA_STATUS_ERROR_OUT_OF_RESOURCES`, and the HIP launch
fallback's reinterpret_cast turned it into a SIGSEGV.
`patches/rocm-systems/d2h-null-dsthost.patch` fixes the second one. A D2H copy
into a host-accessible *device* allocation passed `readBuffer` a NULL host
destination, because `getHostMem()` is not set for memory that came from the
device. Verified on the box with `qwen35_2b_bench_v3.py`: EXIT=0, prefill 311.0
tok/s, decode 30.2 tok/s, on a 2101-token prompt and 128 decode tokens. The
7.14 record for the same bench is 331.7 and 24.4.

CI builds the fork into the published image. `docker/vllm.Dockerfile` wheels it
and compiles its three gfx803 kernels, and `final.Dockerfile` installs both into
`/opt/venv` through `scripts/build/final-vllm.sh`, which also asserts the package
imports. The box-only editable install under `/data/vllm-mobydick/` is the same
source tree and stays for hardware debugging, where a container's PID namespace
gets in the way of gdb and ftrace. Both paths carry the same source, so a result
measured on one is a claim about the other only once the other has been rebuilt.

vLLM in the image runs only if triton can compile an fp16 dot on gfx803, because
the ROCm attention backend's prefill kernel is one. That is
`patches/triton/gfx803-vdot-gate.patch`. Evidence from the box-native stack is
not evidence about the image on this point: the two run different triton builds.
It is also not documented with `.patch.md` files under `patches/vllm/`.
That convention belonged to an older version of this repo that did not vendor
vLLM at all, and commit `a8485b4` ("[Build] Bring working vLLM in") removed it on
purpose. Do not bring it back. Every gfx803 finding in vLLM goes into the real
source files under `vllm/`, the same way every other fix in this repo lands in
real code rather than in a document about a hypothetical patch.

One trap: `vllm/` is nested twice. `vllm/` is the fork's own root, with its own
`AGENTS.md`, `README.md`, and `benchmarks/`. That `vllm/AGENTS.md` is upstream's
contribution policy, about pull requests to `vllm-project/vllm`. It does not
apply here, because this fork is never upstreamed. The importable package is one
level deeper, at `vllm/vllm/`. Real examples are
`vllm/vllm/model_executor/layers/...` and `vllm/vllm/v1/attention/ops/...`. A
write to `vllm/model_executor/...` lands nowhere, or in the wrong file. This is
easy to do when your shell's working directory has drifted, because the tool's
working directory persists between commands in a session. Use an absolute path
when you are unsure.

Hand-written gfx803 HIP kernel source goes in `vllm/vllm/gfx803_kernels/`
(`gfx803_gemm_lib.hip`, `gfx803_attn_split.hip`), a folder of its own, not next
to the Python loaders that `ctypes`-load the compiled output. `vllm/.gitignore`
excludes `*.hip` repo-wide with the comment "hip files generated by PyTorch".
That is true of upstream's hipify output, which is translated from `.cu` CUDA
source during the build, and it is wrong for these two files, which are
hand-written originals with no `.cu` to regenerate them from. The folder is
whitelisted back with `!/vllm/gfx803_kernels/*.hip` in `vllm/.gitignore`, rather
than force-added past the rule each time. Put any new hand-written gfx803 `.hip`
source there, and it tracks normally. The compiled `.so` must still sit next to
its Python loader, for example
`vllm/vllm/model_executor/layers/libgfx803gemm.so`, because the loader builds its
`ctypes` path from `__file__`. Only the source moved. The `hipcc -o` output path
did not move. The loader's docstring gives the exact command.

Box-only work flows back like this. Iterate on a real gfx803 vLLM fix live on the
box at 192.168.1.214 under `/data/vllm-mobydick/` (see "Hardware access" below),
test it there, then copy the changed and new files back into this local
`vllm/vllm/...` checkout, so the real diff lands here. Do not summarize it into
a separate document. The usual rule still applies: build and stage locally, and
do not commit or push unless you are told to for that specific commit.

