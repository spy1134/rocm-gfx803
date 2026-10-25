#!/usr/bin/env python3
"""Choose the hot experts of an Ollama MoE model: the routed experts that the rocm-gfx803 Ollama image keeps in
VRAM while the rest stay in system RAM (patch 140, patches/ollama/llama-cpp-sched/).

    tools/moe_hot_calibrate.py MODEL            e.g. qwen3.6-35b-heretic:32k
    tools/moe_hot_calibrate.py MODEL --cover IMATRIX.gguf
    tools/moe_hot_calibrate.py MODEL --prompts OTHER.json --imatrix-only IMATRIX.gguf

The first form runs the model once per prompt of tools/moe_hot_prompts.json (thinking off, 400 tokens each), formats
each prompt and answer with the model's own chat template, counts with llama-imatrix which experts every layer selects
over those transcripts, and writes the list, most-selected first, to ollama/moe-hot/<model blob>.hot. start-ollama.sh
passes that folder to the server; restart the service to use a new list. The Ollama service must be stopped while
this runs, since the model is loaded on its own. For Qwen3.6 35B-A3B it takes about 10 minutes.

The set has to come from chat: an assistant's answers route to different experts than plain document text, and a list
chosen on articles and source code serves about half as many of a chat's expert reads. Prompts that resemble what
the model is used for give the best list; --prompts takes another file of the same form ({"prompts": [...]}).

--cover IMATRIX.gguf reports which share of the selections counted in another imatrix file the list serves at a few
VRAM budgets: a check of a list against text it was not chosen on.
"""
import argparse, json, os, struct, subprocess, sys, tempfile, time, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8093
NAME = "moe-hot-calibrate"


def model_blob(models_dir, model):
    """Path of the GGUF blob an Ollama model name points to."""
    name, _, tag = model.partition(":")
    parts = name.split("/")
    if len(parts) == 1:
        parts = ["library"] + parts
    manifest = os.path.join(models_dir, "manifests", "registry.ollama.ai", *parts, tag or "latest")
    with open(manifest) as f:
        layers = json.load(f)["layers"]
    digest = next(l["digest"] for l in layers if l["mediaType"] == "application/vnd.ollama.image.model")
    return os.path.join(models_dir, "blobs", digest.replace(":", "-"))


def gpu_args():
    out = subprocess.run(["bash", "-c", f'. "{REPO}/container-args.sh"; gfx803_gpu_args; printf "%s\\n" "${{GFX803_ARGS[@]}}"'],
                         check=True, capture_output=True, text=True).stdout
    return out.splitlines()


def runner(image, blob, work, *args):
    """docker run arguments for one of the image's llama.cpp programs, with the environment Ollama gives its runner."""
    return ["docker", "run", "--rm", *gpu_args(),
            "-v", f"{os.path.dirname(blob)}:/m:ro", "-v", f"{work}:/w",
            "-e", "LD_LIBRARY_PATH=/usr/lib/ollama/rocm_v7_2:/usr/lib/ollama",
            "-e", "GGML_BACKEND_PATH=/usr/lib/ollama/rocm_v7_2/libggml-hip.so",
            "--entrypoint", "/usr/lib/ollama/" + args[0], image, "-m", "/m/" + os.path.basename(blob), *args[1:]]


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def transcripts(image, blob, work, prompts, n_tokens):
    cmd = runner(image, blob, work, "llama-server", "--jinja", "-ngl", "99", "--cpu-moe", "-c", "8192", "-np", "1",
                 "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--host", "0.0.0.0", "--port", "8080")
    cmd[2:2] = ["-d", "--name", NAME, "-p", f"127.0.0.1:{PORT}:8080"]
    cmd.remove("--rm")
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    subprocess.run(cmd, check=True, capture_output=True)
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except OSError:
                if not subprocess.run(["docker", "ps", "-q", "-f", f"name=^{NAME}$"], capture_output=True, text=True).stdout:
                    sys.exit("the server exited:\n" + subprocess.run(["docker", "logs", NAME], capture_output=True, text=True).stderr[-2000:])
                time.sleep(1)
        text = []
        for i, p in enumerate(prompts):
            r = post("/v1/chat/completions", {
                "messages": [{"role": "user", "content": p}], "max_tokens": n_tokens, "temperature": 0.7,
                "top_p": 0.8, "top_k": 20, "seed": 1000 + i, "chat_template_kwargs": {"enable_thinking": False}})
            answer = r["choices"][0]["message"]["content"]
            t = post("/apply-template", {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": answer}],
                                         "chat_template_kwargs": {"enable_thinking": False}})
            text.append(t["prompt"])
            print(f"  {i + 1}/{len(prompts)}: {r['timings']['predicted_n']} tokens at {r['timings']['predicted_per_second']:.1f} tok/s", flush=True)
        return "".join(text)
    finally:
        subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)


def gguf_tensors(path):
    """{name: list of float} for the F32 tensors of a GGUF file (enough for llama-imatrix output)."""
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    def take(fmt):
        nonlocal off
        v = struct.unpack_from("<" + fmt, data, off); off += struct.calcsize("<" + fmt)
        return v[0]
    def string():
        nonlocal off
        n = take("Q"); s = data[off:off + n].decode(); off += n
        return s
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def value(t):
        if t == 8:
            return string()
        if t == 9:
            it, n = take("I"), take("Q")
            return [value(it) for _ in range(n)]
        return take(scalar[t])
    if data[:4] != b"GGUF":
        sys.exit(f"{path} is not a GGUF file")
    off = 4
    take("I")
    n_tensors, n_kv = take("Q"), take("Q")
    align = 32
    for _ in range(n_kv):
        key = string(); v = value(take("I"))
        if key == "general.alignment":
            align = v
    infos = []
    for _ in range(n_tensors):
        name = string(); dims = [take("Q") for _ in range(take("I"))]; typ = take("I"); toff = take("Q")
        infos.append((name, dims, typ, toff))
    base = (off + align - 1) // align * align
    out = {}
    for name, dims, typ, toff in infos:
        if typ == 0:
            n = 1
            for d in dims:
                n *= d
            out[name] = struct.unpack_from(f"<{n}f", data, base + toff)
    return out


def selection_counts(imatrix):
    """{(layer, expert): count} from the per-expert counts llama-imatrix stores for each ffn_down_exps."""
    counts = {}
    for name, c in gguf_tensors(imatrix).items():
        if name.startswith("blk.") and name.endswith(".ffn_down_exps.weight.counts"):
            layer = int(name.split(".")[1])
            for e, v in enumerate(c):
                if v > 0:
                    counts[(layer, e)] = v
    if not counts:
        sys.exit(f"{imatrix} holds no expert counts: is the model a MoE model?")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Ollama model name, e.g. qwen3.6-35b-heretic:32k")
    ap.add_argument("--prompts", default=os.path.join(REPO, "tools", "moe_hot_prompts.json"))
    ap.add_argument("--tokens", type=int, default=400, help="tokens per answer (default 400)")
    ap.add_argument("--image", default="rocm-gfx803-ollama:latest")
    ap.add_argument("--data", default=os.environ.get("OLLAMA_DATA", os.path.join(REPO, "ollama")),
                    help="the Ollama data folder (default ollama/ in this repository)")
    ap.add_argument("--cover", metavar="IMATRIX", help="report the list's share of another imatrix file's selections")
    ap.add_argument("--imatrix-only", metavar="PATH", help="save the counts as an imatrix file at PATH instead of "
                    "writing a list (to check a list against other prompts with --cover)")
    a = ap.parse_args()

    blob = model_blob(os.path.join(a.data, "models"), a.model)
    out = os.path.join(a.data, "moe-hot", os.path.basename(blob) + ".hot")

    if a.cover:
        hot = [tuple(map(int, l.split()[:2])) for l in open(out) if l[:1].isdigit()]
        other = selection_counts(a.cover)
        total = sum(other.values())
        for n in (1024, 2048, 3072, len(hot)):
            print(f"{min(n, len(hot)):5d} hottest: {100 * sum(other.get(k, 0) for k in hot[:n]) / total:5.1f}% of the selections")
        return

    if subprocess.run(["docker", "ps", "-q", "-f", "name=^ollama$"], capture_output=True, text=True).stdout:
        sys.exit("the Ollama service is running; stop it first (./start-ollama.sh --stop)")
    prompts = json.load(open(a.prompts))["prompts"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(out)) as work:
        os.chmod(work, 0o777)
        print(f"generating {len(prompts)} answers with {a.model}", flush=True)
        with open(os.path.join(work, "transcripts.txt"), "w") as f:
            f.write(transcripts(a.image, blob, work, prompts, a.tokens))
        print("counting expert selections (llama-imatrix)", flush=True)
        r = subprocess.run(runner(a.image, blob, work, "llama-imatrix", "-f", "/w/transcripts.txt", "-o", "/w/imatrix.gguf",
                                  "--parse-special", "-c", "512", "-b", "512", "-ngl", "99", "--cpu-moe", "-fa", "on"),
                           capture_output=True, text=True)
        if r.returncode or not os.path.exists(os.path.join(work, "imatrix.gguf")):
            sys.exit("llama-imatrix failed:\n" + r.stderr[-2000:])
        counts = selection_counts(os.path.join(work, "imatrix.gguf"))
        if a.imatrix_only:
            os.replace(os.path.join(work, "imatrix.gguf"), a.imatrix_only)
            print(f"wrote {a.imatrix_only}")
            return
    with open(out + ".tmp", "w") as f:
        f.write(f"# hot experts of {a.model}: layer expert count, most-selected first; {len(prompts)} chat transcripts\n")
        for (layer, e), c in sorted(counts.items(), key=lambda kv: -kv[1]):
            f.write(f"{layer} {e} {c:.0f}\n")
    os.replace(out + ".tmp", out)
    print(f"wrote {out} ({len(counts)} experts); restart the Ollama service to use it")


if __name__ == "__main__":
    main()
