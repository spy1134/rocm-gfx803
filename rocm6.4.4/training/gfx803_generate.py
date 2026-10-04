"""SD 1.5 text-to-image through ComfyUI's HTTP API, for gfx803-app.

Builds a standard SD 1.5 workflow from a settings dict, queues it on the
local ComfyUI, and reads results back from its output folder. Each image
carries the settings that made it in its PNG metadata (key "gfx803"), so the
gallery and "requeue with these settings" work across ComfyUI restarts and
for images opened anywhere else.

Presets are named settings dicts in $WORKSPACE/generate/presets.json.
"""
import json
import os
import random
import time
import urllib.error
import urllib.request
import uuid

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
GEN_DIR = os.path.join(WORKSPACE, "generate")
PRESETS = os.path.join(GEN_DIR, "presets.json")
LAST = os.path.join(GEN_DIR, "last.json")
EXTRA_PATHS = os.path.join(GEN_DIR, "extra_model_paths.yaml")
COMFY_URL = "http://127.0.0.1:8188"
MODELS = "/opt/ComfyUI/models"
OUTPUT = "/opt/ComfyUI/output"
OUT_PREFIX = "gfx803"
TRAINING_OUTPUT = os.path.join(WORKSPACE, "output")
# Training checkpoints show up in ComfyUI's LoRA list under this folder name.
TRAINING_LORA_ROOT = "training"

SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast",
            "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde",
            "ddim", "uni_pc"]
SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"]
SIZES = [(512, 512), (512, 768), (768, 512), (512, 640), (640, 512), (448, 576), (576, 448)]

DEFAULTS = {
    "checkpoint": "", "vae": "", "loras": [], "clip_skip": 1,
    "positive": "", "negative": "lowres, bad anatomy, bad hands, worst quality, low quality, watermark, artist name",
    "width": 512, "height": 512, "steps": 25, "cfg": 7.0, "sampler": "dpmpp_2m", "scheduler": "karras",
    "seed": -1, "count": 1,
}


def extra_model_paths_yaml():
    """ComfyUI config that adds training output as a LoRA folder, so training
    checkpoints can be compared without copying them into models/loras."""
    os.makedirs(GEN_DIR, exist_ok=True)
    with open(EXTRA_PATHS, "w") as f:
        f.write(f"{TRAINING_LORA_ROOT}:\n  base_path: {WORKSPACE}\n  loras: output/\n")
    return EXTRA_PATHS


# ------------------------------------------------------------------ models
def model_files(kind):
    """Model names as ComfyUI expects them (paths relative to models/<kind>)."""
    root = os.path.join(MODELS, kind)
    out = []
    for dirpath, _dirs, files in os.walk(root, followlinks=True):
        for f in files:
            if f.lower().endswith((".safetensors", ".ckpt", ".pt")):
                out.append(os.path.relpath(os.path.join(dirpath, f), root))
    return sorted(out, key=str.lower)


def training_loras():
    """{run: [(step or None, name as ComfyUI resolves it)]} for training
    checkpoints, reachable through extra_model_paths_yaml()."""
    import gfx803_status as st
    out = {}
    try:
        runs = sorted(os.listdir(TRAINING_OUTPUT))
    except OSError:
        return out
    for run in runs:
        cks = st.checkpoints(run)
        if cks:
            out[run] = [(step, os.path.relpath(path, TRAINING_OUTPUT)) for step, path in cks]
    return out


# ------------------------------------------------------------------ workflow
def build_workflow(s, seed):
    """API-format ComfyUI graph for one image of settings s."""
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": s["checkpoint"]}}}
    model, clip = ["1", 0], ["1", 1]
    for i, lora in enumerate(l for l in s.get("loras", []) if l.get("name")):
        nid = str(20 + i)
        wf[nid] = {"class_type": "LoraLoader",
                   "inputs": {"lora_name": lora["name"], "strength_model": float(lora.get("strength", 1.0)),
                              "strength_clip": float(lora.get("strength", 1.0)), "model": model, "clip": clip}}
        model, clip = [nid, 0], [nid, 1]
    wf["3"] = {"class_type": "CLIPSetLastLayer", "inputs": {"stop_at_clip_layer": -int(s.get("clip_skip", 1)), "clip": clip}}
    wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": s["positive"], "clip": ["3", 0]}}
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": s.get("negative", ""), "clip": ["3", 0]}}
    wf["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": int(s["width"]), "height": int(s["height"]), "batch_size": 1}}
    wf["7"] = {"class_type": "KSampler",
               "inputs": {"seed": int(seed), "steps": int(s["steps"]), "cfg": float(s["cfg"]), "sampler_name": s["sampler"],
                          "scheduler": s["scheduler"], "denoise": 1.0, "model": model, "positive": ["4", 0],
                          "negative": ["5", 0], "latent_image": ["6", 0]}}
    vae = ["1", 2]
    if s.get("vae"):
        wf["8"] = {"class_type": "VAELoader", "inputs": {"vae_name": s["vae"]}}
        vae = ["8", 0]
    wf["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": vae}}
    wf["2"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": f"{OUT_PREFIX}/{OUT_PREFIX}", "images": ["9", 0]}}
    return wf


def resolve_seed(seed):
    seed = int(seed)
    return random.randint(0, 2**32 - 1) if seed < 0 else seed


# ------------------------------------------------------------------ API
class ComfyError(Exception):
    pass


def _request(path, data=None, timeout=10):
    req = urllib.request.Request(COMFY_URL + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            j = json.loads(detail)
            msg = j.get("error", {}).get("message", "") or detail
            for node in (j.get("node_errors") or {}).values():
                for err in node.get("errors", []):
                    msg += f"; {err.get('message', '')}: {err.get('details', '')}"
            detail = msg
        except ValueError:
            pass
        raise ComfyError(detail[:500]) from None
    except OSError as e:
        raise ComfyError(f"ComfyUI not reachable ({e})") from None
    return json.loads(body) if body else {}


def available():
    try:
        _request("/system_stats", timeout=2)
        return True
    except ComfyError:
        return False


def submit(s, seed, label="", group=None, client_id=None):
    """Queues one image; returns the prompt id."""
    meta = dict(s, seed=seed, label=label, group=group, queued=time.time())
    body = {"prompt": build_workflow(s, seed), "client_id": client_id or "gfx803-app",
            "extra_data": {"extra_pnginfo": {"gfx803": meta}}}
    return _request("/prompt", body)["prompt_id"]


def queue_state():
    """(running, pending) as lists of (prompt id, gfx803 settings or None)."""
    q = _request("/queue")

    def items(entries):
        # Each entry is [number, prompt_id, prompt, extra_data, outputs]; the
        # number is the queue position ComfyUI runs them in.
        out = []
        for e in sorted(entries, key=lambda e: e[0]):
            meta = (e[3] or {}).get("extra_pnginfo", {}).get("gfx803") if len(e) > 3 else None
            out.append((e[1], meta))
        return out

    return items(q.get("queue_running", [])), items(q.get("queue_pending", []))


def cancel(prompt_id, running=False):
    if running:
        _request("/interrupt", {})
    else:
        _request("/queue", {"delete": [prompt_id]})


def object_info_choices():
    """(samplers, schedulers) from ComfyUI itself when it is running."""
    try:
        info = _request("/object_info/KSampler")["KSampler"]["input"]["required"]
        return list(info["sampler_name"][0]), list(info["scheduler"][0])
    except (ComfyError, KeyError, IndexError, TypeError):
        return SAMPLERS, SCHEDULERS


# ------------------------------------------------------------------ results
def image_settings(path):
    """The gfx803 settings stored in a generated PNG, or None."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            raw = im.info.get("gfx803")
        return json.loads(raw) if raw else None
    except (OSError, ValueError):
        return None


def gallery(limit=400):
    """Generated images, newest first, as (path, settings)."""
    d = os.path.join(OUTPUT, OUT_PREFIX)
    try:
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".png")]
    except OSError:
        return []
    files.sort(key=os.path.getmtime, reverse=True)
    return [(f, None) for f in files[:limit]]


# ------------------------------------------------------------------ presets
def _load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=1)
    os.replace(tmp, path)


def presets():
    return _load_json(PRESETS, {})


def save_preset(name, settings):
    p = presets()
    p[name] = settings
    _save_json(PRESETS, p)


def delete_preset(name):
    p = presets()
    p.pop(name, None)
    _save_json(PRESETS, p)


def last_settings():
    return dict(DEFAULTS, **_load_json(LAST, {}))


def save_last(settings):
    _save_json(LAST, settings)


def new_group():
    return uuid.uuid4().hex[:8]
