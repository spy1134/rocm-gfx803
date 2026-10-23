#!/usr/bin/env python3
"""
gfx803 correctness check for sd-scripts LoRA training.

Compares GPU results against a CPU reference for the operations an SD1.5
UNet uses in training, then for the whole UNet. Run inside the container,
with ComfyUI idle (no model loaded) so the card's VRAM is free:

    python /opt/gfx803-training/gfx803_unet_check.py \
        --ckpt /opt/ComfyUI/models/checkpoints/SD15/v1-5-pruned-emaonly.safetensors

Stages:
  1. Ops    - fp32 attention (baddbmm/softmax/bmm) at the UNet's real shapes,
              linear, conv2d, group_norm, layer_norm (via the shim). Forward
              AND backward, several reps each with fresh random data, GPU vs
              CPU float64.
  2. UNet   - full UNet forward at 512x512 (64x64 latent), GPU vs CPU, plus the
              MSE loss against the noise, at three timesteps. GPU runs several
              times per input because these bugs often show up only after the
              first call.
  3. Bisect - runs automatically if stage 2 fails (or with --bisect). Hooks
              every leaf module and prints the first places where GPU and CPU
              disagree. "in  X" diverging first means the damage happened in
              functional code between modules (e.g. the attention matmuls);
              "out X" diverging with "in X" fine means module X itself.
              Note: the hooks copy tensors to the CPU, which adds syncs; if
              stage 2 fails but stage 3 finds nothing, the hooks changed the
              timing enough to hide the bug.

Tolerance follows KERNEL_BUGS.md: an element is bad only if
abs_err > 0.05 AND rel_err > 1e-2.
"""
import argparse
import copy
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/opt/sd-scripts")

ABS_TOL = 5e-2
REL_TOL = 1e-2


def diff(gpu, ref):
    """Return (max_abs_err, mean_abs_err, n_bad, n_nonfinite, numel)."""
    a = gpu.detach().double().cpu()
    b = ref.detach().double().cpu()
    nonfinite = (~torch.isfinite(a)).sum().item()
    err = torch.nan_to_num((a - b).abs(), nan=float("inf"), posinf=float("inf"))
    rel = err / b.abs().clamp_min(1e-6)
    bad = ((err > ABS_TOL) & (rel > REL_TOL)).sum().item()
    finite_err = err[torch.isfinite(err)]
    mean = finite_err.mean().item() if finite_err.numel() else float("inf")
    return err.max().item(), mean, bad, nonfinite, err.numel()


# ---------------------------------------------------------------- stage 1: ops

def make_cases(gen):
    def rn(*s, scale=1.0):
        return torch.randn(*s, generator=gen) * scale

    def attn(bh, sq, sk, d):
        # sd-scripts' "normal" attention: baddbmm(beta=0) -> softmax -> bmm
        def fn(q, k, v):
            s = torch.baddbmm(
                torch.empty(bh, sq, sk, dtype=q.dtype, device=q.device),
                q, k.transpose(1, 2), beta=0, alpha=d ** -0.5)
            return torch.bmm(s.softmax(dim=-1), v)
        return fn, lambda: (rn(bh, sq, d), rn(bh, sk, d), rn(bh, sk, d))

    def conv(cin, cout, k, hw, stride=1):
        pad = k // 2
        fn = lambda x, w, b: F.conv2d(x, w, b, stride=stride, padding=pad)
        mk = lambda: (rn(1, cin, hw, hw), rn(cout, cin, k, k, scale=(cin * k * k) ** -0.5), rn(cout))
        return fn, mk

    def linear(n, cin, cout):
        fn = lambda x, w, b: F.linear(x, w, b)
        mk = lambda: (rn(1, n, cin), rn(cout, cin, scale=cin ** -0.5), rn(cout))
        return fn, mk

    return {
        "attn self  64x64 lvl (8x4096x4096, d40)": attn(8, 4096, 4096, 40),
        "attn cross 64x64 lvl (8x4096x77,   d40)": attn(8, 4096, 77, 40),
        "attn self  32x32 lvl (8x1024x1024, d80)": attn(8, 1024, 1024, 80),
        "attn self  16x16 lvl (8x256x256,  d160)": attn(8, 256, 256, 160),
        "attn self   8x8  lvl (8x64x64,    d160)": attn(8, 64, 64, 160),
        "linear 4096x320 -> 320":                  linear(4096, 320, 320),
        "linear 4096x320 -> 2560 (GEGLU)":         linear(4096, 320, 2560),
        "linear 4096x320 -> 8 (LoRA down)":        linear(4096, 320, 8),
        "linear 4096x8 -> 320 (LoRA up)":          linear(4096, 8, 320),
        "conv 3x3 320->320 @64":                   conv(320, 320, 3, 64),
        "conv 3x3 640->640 @32":                   conv(640, 640, 3, 32),
        "conv 3x3 320->320 @64 stride2":           conv(320, 320, 3, 64, stride=2),
        "conv 1x1 320->320 @64":                   conv(320, 320, 1, 64),
        "conv 3x3 4->320 @64 (conv_in)":           conv(4, 320, 3, 64),
        "group_norm 32g 320x64x64": (
            lambda x, w, b: F.group_norm(x, 32, w, b, 1e-5),
            lambda: (rn(1, 320, 64, 64), rn(320), rn(320))),
        "layer_norm 4096x320 (shim on GPU)": (
            lambda x, w, b: F.layer_norm(x, (320,), w, b, 1e-5),
            lambda: (rn(1, 4096, 320, scale=3.0), rn(320), rn(320))),
    }


def run_ops(reps, backward, only=None, no_empty_cache=False):
    print(f"\n=== Stage 1: ops ({reps} reps, backward={'on' if backward else 'off'}"
          f"{', only=' + repr(only) if only else ''}) ===")
    gen = torch.Generator().manual_seed(1234)
    failed = 0
    for name, (fn, make) in make_cases(gen).items():
        if only and only.lower() not in name.lower():
            continue
        print(f"[....] {name}", flush=True)
        worst, bad_reps, notes = 0.0, 0, []
        for r in range(reps):
            inputs = make()
            ref_in = [x.double().requires_grad_(backward) for x in inputs]
            gpu_in = [x.cuda().requires_grad_(backward) for x in inputs]
            ref_out = fn(*ref_in)
            print(f"       rep {r}: GPU forward", flush=True)
            gpu_out = fn(*gpu_in)
            torch.cuda.synchronize()
            checks = [("out", gpu_out, ref_out)]
            if backward:
                go = torch.randn(ref_out.shape, generator=gen)
                (ref_out * go.double()).sum().backward()
                print(f"       rep {r}: GPU backward", flush=True)
                (gpu_out * go.cuda()).sum().backward()
                torch.cuda.synchronize()
                checks += [(f"grad[{i}]", g.grad, c.grad)
                           for i, (g, c) in enumerate(zip(gpu_in, ref_in))]
            rep_bad = False
            for label, g, c in checks:
                mx, mean, bad, nf, n = diff(g, c)
                worst = max(worst, mx)
                if bad or nf:
                    rep_bad = True
                    notes.append(f"rep {r} {label}: max_abs={mx:.3e} mean_abs={mean:.3e} "
                                 f"bad={bad}/{n} nonfinite={nf}")
            bad_reps += rep_bad
            del ref_in, gpu_in, ref_out, gpu_out, checks
        status = "OK  " if bad_reps == 0 else "FAIL"
        print(f"[{status}] {name}: {bad_reps}/{reps} reps bad, worst max_abs={worst:.3e}")
        for note in notes[:4]:
            print("         " + note)
        failed += bad_reps > 0
        if not no_empty_cache:
            torch.cuda.empty_cache()
    print(f"Stage 1: {failed} op(s) failing")
    return failed


# --------------------------------------------------------------- stage 2: UNet

def load_unet(ckpt, attn):
    from library import model_util
    te, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(False, ckpt)
    del te, vae
    unet = unet.float().eval()
    if hasattr(unet, "set_use_memory_efficient_attention"):
        unet.set_use_memory_efficient_attention(False, attn == "mem_eff")
    if hasattr(unet, "set_use_sdpa"):
        unet.set_use_sdpa(attn == "sdpa")
    return unet


def unet_call(unet, x, t, c):
    out = unet(x, t, c)
    if hasattr(out, "sample"):
        out = out.sample
    elif isinstance(out, (tuple, list)):
        out = out[0]
    return out


def make_unet_inputs(ctx_len, gen, timestep):
    x0 = torch.randn(1, 4, 64, 64, generator=gen)
    noise = torch.randn(1, 4, 64, 64, generator=gen)
    # SD1.5 scaled_linear schedule
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
    ac = torch.cumprod(1 - betas, 0)[timestep]
    noisy = (ac.sqrt() * x0 + (1 - ac).sqrt() * noise).float()
    cond = torch.randn(1, ctx_len, 768, generator=gen)
    t = torch.tensor([timestep], dtype=torch.long)
    return noisy, t, cond, noise


def run_unet(unet_cpu, unet_gpu, ctx_len, gpu_reps):
    print(f"\n=== Stage 2: full UNet forward (64x64 latent, ctx={ctx_len}, "
          f"{gpu_reps} GPU runs per input) ===")
    gen = torch.Generator().manual_seed(42)
    failed = 0
    for ts in (100, 500, 900):
        noisy, t, cond, noise = make_unet_inputs(ctx_len, gen, ts)
        with torch.no_grad():
            ref = unet_call(unet_cpu, noisy, t, cond)
            ref_loss = F.mse_loss(ref, noise).item()
            print(f"t={ts}: CPU loss={ref_loss:.5f}")
            for r in range(gpu_reps):
                out = unet_call(unet_gpu, noisy.cuda(), t.cuda(), cond.cuda())
                loss = F.mse_loss(out.float().cpu(), noise).item()
                mx, mean, bad, nf, n = diff(out, ref)
                ok = bad == 0 and nf == 0
                failed += not ok
                print(f"  [{'OK  ' if ok else 'FAIL'}] GPU run {r}: loss={loss:.5f} "
                      f"max_abs={mx:.3e} mean_abs={mean:.3e} bad={bad}/{n} nonfinite={nf}")
                del out
        torch.cuda.empty_cache()
    print(f"Stage 2: {failed} failing GPU run(s)")
    return failed


# ------------------------------------------------------------- stage 3: bisect

def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for y in x:
            if torch.is_tensor(y):
                return y
    return None


def _attach_hooks(unet):
    log, hooks = [], []
    for name, mod in unet.named_modules():
        if any(True for _ in mod.children()):
            continue  # leaf modules only

        def pre(m, inp, name=name):
            t_ = _first_tensor(inp)
            if t_ is not None:
                log.append(("in ", name, t_.detach().float().cpu().clone()))

        def post(m, inp, out, name=name):
            t_ = _first_tensor(out)
            if t_ is not None:
                log.append(("out", name, t_.detach().float().cpu().clone()))

        hooks.append(mod.register_forward_pre_hook(pre))
        hooks.append(mod.register_forward_hook(post))
    return log, hooks


def run_bisect(unet_cpu, unet_gpu, ctx_len, max_show=10):
    print("\n=== Stage 3: bisect (first divergent leaf modules, in execution order) ===")
    gen = torch.Generator().manual_seed(7)
    noisy, t, cond, _ = make_unet_inputs(ctx_len, gen, 500)

    log_c, hc = _attach_hooks(unet_cpu)
    with torch.no_grad():
        unet_call(unet_cpu, noisy, t, cond)
    for h in hc:
        h.remove()

    log_g, hg = _attach_hooks(unet_gpu)
    with torch.no_grad():
        unet_call(unet_gpu, noisy.cuda(), t.cuda(), cond.cuda())
    for h in hg:
        h.remove()

    if len(log_c) != len(log_g):
        print(f"warning: CPU logged {len(log_c)} tensors, GPU {len(log_g)}")
    shown = 0
    for (kc, nc, tc), (kg, ng, tg) in zip(log_c, log_g):
        if (kc, nc) != (kg, ng):
            print(f"execution order differs at {kc}{nc} vs {kg}{ng}; stopping")
            break
        if tc.shape != tg.shape:
            continue
        mx, mean, bad, nf, n = diff(tg, tc)
        if bad or nf:
            print(f"  {kg} {ng}: max_abs={mx:.3e} mean_abs={mean:.3e} bad={bad}/{n} nonfinite={nf}")
            shown += 1
            if shown >= max_show:
                break
    if shown == 0:
        print("  no divergence found with hooks attached")


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="SD1.5 checkpoint; omit to run only the op tests")
    ap.add_argument("--reps", type=int, default=5, help="reps per op test")
    ap.add_argument("--unet-reps", type=int, default=3, help="GPU runs per UNet input")
    ap.add_argument("--ctx", type=int, default=77,
                    help="text context length (sd-scripts uses 227 with --max_token_length=225)")
    ap.add_argument("--attn", choices=["normal", "sdpa", "mem_eff"], default="normal",
                    help="UNet attention implementation (sd-scripts default is normal)")
    ap.add_argument("--skip-ops", action="store_true")
    ap.add_argument("--no-backward", action="store_true")
    ap.add_argument("--bisect", action="store_true", help="always run stage 3")
    ap.add_argument("--only", help="run only op tests whose name contains this text, e.g. 'GEGLU' or '8x8'")
    ap.add_argument("--no-empty-cache", action="store_true",
                    help="don't release cached GPU memory between op tests")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 8)
    print(f"torch {torch.__version__}, HIP {torch.version.hip}")
    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    for var in ("LD_PRELOAD", "GFX803_SGEMM_SHIM_DISABLE", "HSA_DISABLE_FRAGMENT_ALLOCATOR",
                "AMD_SERIALIZE_KERNEL", "HIP_LAUNCH_BLOCKING", "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_HIP_ALLOC_CONF"):
        print(f"  {var}={os.environ.get(var, '<unset>')}")
    if not torch.cuda.is_available():
        sys.exit("No GPU visible to torch.")
    # Show which runtime libraries were really loaded (a replaced file that
    # isn't the one the soname resolves to is an easy way to test the wrong lib).
    torch.zeros(1, device="cuda")
    try:
        with open("/proc/self/maps") as maps:
            libs = sorted({line.split()[-1] for line in maps
                           if any(s in line for s in ("libhsa-runtime64", "sgemm_shim", "librocblas.so"))})
        for lib in libs:
            print(f"  loaded: {lib}")
    except OSError:
        pass

    if not args.skip_ops:
        run_ops(args.reps, not args.no_backward, args.only, args.no_empty_cache)

    if args.ckpt:
        print(f"\nLoading UNet from {args.ckpt} (attention: {args.attn})...")
        unet_cpu = load_unet(args.ckpt, args.attn)
        unet_gpu = copy.deepcopy(unet_cpu).cuda()
        failed = run_unet(unet_cpu, unet_gpu, args.ctx, args.unet_reps)
        if failed or args.bisect:
            run_bisect(unet_cpu, unet_gpu, args.ctx)


if __name__ == "__main__":
    main()
