"""SD1.5 UNet step (512 px, CFG batch 2) broken down by operation type with CUDA events around leaf modules.

Run in the ComfyUI image: the gfx803 attention node is imported from /tools the way ComfyUI loads it. Exclusive time of the
attention module (its projections are Linear leaves) is the attention core. 'other' is the step total minus every recorded
leaf: residual adds, reshapes, timestep embedding maths, launch gaps."""
import importlib.util, os, sys, time, collections
import torch, torch.nn as nn
sys.path.insert(0, "/opt/ComfyUI"); sys.path.insert(0, "/tools")
import comfy.sd, comfy.model_management
import comfy.ldm.modules.attention as ca
if os.environ.get("NODE", "1") == "1":
    spec = importlib.util.spec_from_file_location("gfx803node", "/tools/ComfyUI-gfx803-attention/__init__.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    print("attention node loaded")
CKPT = "/opt/ComfyUI/models/checkpoints/SD15/v1-5-pruned-emaonly.safetensors"
model = comfy.sd.load_checkpoint_guess_config(CKPT)[0]
comfy.model_management.load_models_gpu([model])
bm = model.model
B = 2
x = torch.randn(B, 4, 64, 64, device="cuda"); sigma = torch.tensor([500.0] * B, device="cuda")
ctx = torch.randn(B, 77, 768, device="cuda", dtype=bm.get_dtype_inference())
def step():
    with torch.no_grad(): return bm.apply_model(x, sigma, c_crossattn=ctx)
for _ in range(3): step()
torch.cuda.synchronize()
REPS = int(os.environ.get("REPS", "5"))
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
for _ in range(REPS): step()
e1.record(); torch.cuda.synchronize()
total = e0.elapsed_time(e1) / REPS
print(f"step without hooks: {total:.1f} ms  (dtype {bm.get_dtype_inference()})")

CAT = [(nn.Conv2d, "conv"), (nn.Linear, "linear"), (nn.GroupNorm, "norm"), (nn.LayerNorm, "norm"),
       (nn.SiLU, "act"), (nn.GELU, "act"), (nn.Dropout, "act"), (ca.CrossAttention, "attention")]
def cat_of(mod):
    for c, n in CAT:
        if isinstance(mod, c): return n
    return None
recs = []; stack = []
def pre(mod, inp):
    ev = torch.cuda.Event(enable_timing=True); ev.record()
    recs.append([cat_of(mod), type(mod).__name__, ev, None, stack[-1] if stack else None]); stack.append(len(recs) - 1)
def post(mod, inp, out):
    ev = torch.cuda.Event(enable_timing=True); ev.record()
    i = stack.pop(); recs[i][3] = ev
hs = []
for mod in bm.diffusion_model.modules():
    if cat_of(mod):
        hs.append(mod.register_forward_pre_hook(pre)); hs.append(mod.register_forward_hook(post))
for _ in range(2): step()
torch.cuda.synchronize(); recs.clear()
for _ in range(REPS): step()
torch.cuda.synchronize()
incl = [r[2].elapsed_time(r[3]) for r in recs]
excl = list(incl)
for i, r in enumerate(recs):
    if r[4] is not None: excl[r[4]] -= incl[i]
agg = collections.defaultdict(float); cnt = collections.Counter(); byname = collections.defaultdict(float)
for r, e in zip(recs, excl):
    agg[r[0]] += e / REPS; cnt[r[0]] += 1; byname[r[1]] += e / REPS
hooked = sum(incl[i] for i, r in enumerate(recs) if r[4] is None) / REPS
print(f"hooked step (events add overhead): {hooked:.1f} ms of modules; modules per step: {len(recs)//REPS}")
print(f"{'category':12} {'ms/step':>8} {'share':>7}  calls/step")
for k in ("conv", "linear", "attention", "norm", "act"):
    print(f"{k:12} {agg[k]:8.1f} {agg[k]/total*100:6.1f}%  {cnt[k]//REPS}")
other = total - sum(agg.values())
print(f"{'other':12} {other:8.1f} {other/total*100:6.1f}%  (residuals, reshapes, launch gaps; estimate)")
print("by class:", {k: round(v, 1) for k, v in sorted(byname.items(), key=lambda kv: -kv[1])[:8]})
