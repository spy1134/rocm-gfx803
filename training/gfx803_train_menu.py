"""Interactive front end for gfx803-train.sh.

Usage: gfx803_train_menu.py OUTFILE

Shows menus for a new run (dataset, base model, settings) or for resuming an
unfinished one, then writes the chosen gfx803-train.sh arguments to OUTFILE,
NUL-separated, and exits 0. Exits 1 if the user cancels. The wrapper parses
those arguments exactly as if they had been typed, so every check it makes
on the command line applies to menu choices too.

Paths and defaults come from the environment the wrapper sets: DATA_ROOT,
CKPT_DIR, VAE_DIR, RUNS_DIR, OUT_ROOT, DEFAULT_MODEL, DEFAULT_VAE, STEPS, LR,
DIM, ALPHA, SAVE_EVERY, RESOLUTION, MIN_BUCKET, MAX_BUCKET.
"""
import curses
import os
import re
import shlex
import sys

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
MODEL_EXT = (".safetensors", ".ckpt")
VAE_EXT = (".safetensors", ".ckpt", ".pt")


def env(name, default=""):
    return os.environ.get(name, default)


# ------------------------------------------------------------------ data
def datasets(root):
    out = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        images = captions = 0
        concepts = []
        for sub in sorted(os.listdir(path)):
            sp = os.path.join(path, sub)
            if not (os.path.isdir(sp) and re.match(r"^\d+_", sub)):
                continue
            concepts.append(sub)
            for f in os.listdir(sp):
                stem, ext = os.path.splitext(f)
                if ext.lower() in IMAGE_EXT:
                    images += 1
                    captions += os.path.exists(os.path.join(sp, stem + ".txt"))
        out.append({"name": name, "images": images, "captions": captions, "concepts": concepts})
    return out


def model_files(root, exts):
    found = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                found.append(os.path.relpath(os.path.join(dirpath, f), root))
    return sorted(found, key=str.lower)


def default_index(items, wanted):
    """Index of the item named like the wrapper's default: exact name first,
    so SD15/v1-5-pruned-emaonly picks that file over a -fp16 variant."""
    wanted = wanted.lower()
    stems = [os.path.splitext(item)[0].lower() for item in items]
    for i, stem in enumerate(stems):
        if wanted and (stem == wanted or os.path.basename(stem) == wanted):
            return i
    for i, item in enumerate(items):
        if wanted and wanted in item.lower():
            return i
    return 0


def read_kv(path):
    kv = {}
    try:
        with open(path) as f:
            for line in f:
                key, sep, value = line.rstrip("\n").partition("=")
                if sep:
                    kv[key] = value
    except OSError:
        pass
    return kv


def unfinished_runs(runs_dir, out_root):
    """Runs gfx803-train.sh would offer to resume, newest first.

    A run still marked running may belong to a live training process in
    another container; it is listed with that warning, and the wrapper's own
    resume checks still apply.
    """
    runs = []
    try:
        names = os.listdir(runs_dir)
    except OSError:
        return runs
    for name in names:
        d = os.path.join(runs_dir, name)
        if not os.path.isfile(os.path.join(d, "run.env")):
            continue
        status = read_kv(os.path.join(d, "status"))
        state = status.get("STATE", "")
        if state not in ("failed", "interrupted", "nan", "running"):
            continue
        total = read_kv(os.path.join(d, "run.env")).get("STEPS", "?")
        done = 0
        for f in os.listdir(os.path.join(out_root, name)) if os.path.isdir(os.path.join(out_root, name)) else []:
            m = re.search(r"-step(\d{8})-state$", f)
            if m:
                done = max(done, int(m.group(1)))
        runs.append({"name": name, "state": state, "done": done, "total": total,
                     "updated": int(status.get("UPDATED", "0") or 0)})
    runs.sort(key=lambda r: r["updated"], reverse=True)
    return runs


# ------------------------------------------------------------------ UI
class Quit(Exception):
    pass


class Menu:
    def __init__(self, scr):
        self.scr = scr
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        curses.use_default_colors()
        self.color = curses.has_colors()
        if self.color:
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)

    def attr(self, pair, extra=0):
        return (curses.color_pair(pair) if self.color else 0) | extra

    def put(self, y, x, text, attr=0):
        h, w = self.scr.getmaxyx()
        if 0 <= y < h and x < w:
            try:
                self.scr.addnstr(y, x, text, max(w - x - 1, 0), attr)
            except curses.error:
                pass

    def frame(self, title, footer):
        self.scr.erase()
        self.put(0, 1, "gfx803-train", self.attr(1, curses.A_BOLD))
        self.put(0, 15, "- " + title, curses.A_BOLD)
        h, _ = self.scr.getmaxyx()
        self.put(h - 1, 1, footer, curses.A_DIM)

    def pick(self, title, items, start=0, describe=None, footer=None):
        """Arrow-key list. Returns the index chosen, or None for Esc/q."""
        if not items:
            return None
        sel, top = min(start, len(items) - 1), 0
        while True:
            h, _ = self.scr.getmaxyx()
            rows = max(h - 5, 1)
            top = min(max(top, sel - rows + 1), sel)
            self.frame(title, footer or "Up/Down select   Enter choose   Esc back   q quit")
            for i, item in enumerate(items[top:top + rows]):
                idx = top + i
                mark = "> " if idx == sel else "  "
                self.put(2 + i, 2, mark + item, curses.A_REVERSE if idx == sel else 0)
            if describe:
                self.put(h - 2, 2, describe(sel), self.attr(2))
            key = self.scr.getch()
            if key in (curses.KEY_UP, ord("k")):
                sel = (sel - 1) % len(items)
            elif key in (curses.KEY_DOWN, ord("j")):
                sel = (sel + 1) % len(items)
            elif key == curses.KEY_PPAGE:
                sel = max(sel - rows, 0)
            elif key == curses.KEY_NPAGE:
                sel = min(sel + rows, len(items) - 1)
            elif key in (curses.KEY_ENTER, 10, 13):
                return sel
            elif key == 27:
                return None
            elif key == ord("q"):
                raise Quit()

    def edit(self, y, x, value, width):
        """Single-line text input at (y, x). Returns the new text, or None for Esc."""
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        buf, pos = list(value), len(value)
        try:
            while True:
                self.put(y, x, " " * width)
                self.put(y, x, "".join(buf)[-width + 1:], curses.A_UNDERLINE)
                self.scr.move(y, x + min(pos, width - 1))
                key = self.scr.get_wch()
                if key in ("\n", "\r", curses.KEY_ENTER):
                    return "".join(buf).strip()
                if key == "\x1b":
                    return None
                if key in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                    if pos:
                        del buf[pos - 1]
                        pos -= 1
                elif key == curses.KEY_DC:
                    if pos < len(buf):
                        del buf[pos]
                elif key == curses.KEY_LEFT:
                    pos = max(pos - 1, 0)
                elif key == curses.KEY_RIGHT:
                    pos = min(pos + 1, len(buf))
                elif key == curses.KEY_HOME:
                    pos = 0
                elif key == curses.KEY_END:
                    pos = len(buf)
                elif isinstance(key, str) and key.isprintable():
                    buf.insert(pos, key)
                    pos += 1
        finally:
            try:
                curses.curs_set(0)
            except curses.error:
                pass


# ------------------------------------------------------------------ forms
def new_run_form(m, sets, models, vaes, actions=("Start training", "Dry run (show the command)", "Cancel")):
    """Returns the gfx803-train arguments for the chosen settings, or None."""
    def field(key, label, kind, value, options=None, check=None, help_=""):
        return {"key": key, "label": label, "kind": kind, "value": value,
                "options": options or [], "check": check, "help": help_}

    is_int = lambda v: re.fullmatch(r"\d+", v) is not None and int(v) > 0
    is_lr = lambda v: re.fullmatch(r"\d*\.?\d+(e-?\d+)?", v, re.I) is not None
    is_buckets = lambda v: re.fullmatch(r"\d+:\d+", v) is not None
    is_mib = lambda v: re.fullmatch(r"\d+", v) is not None

    set_labels = [s["name"] for s in sets]
    vae_choices = ["(default: %s)" % env("DEFAULT_VAE"), "none (use the checkpoint's)"] + vaes
    fields = [
        field("data", "Dataset", "pick", 0, set_labels, help_="sd-scripts folder of <repeats>_<trigger>/ subfolders"),
        field("model", "Base model", "pick", default_index(models, env("DEFAULT_MODEL")), models),
        field("vae", "VAE", "pick", 0, vae_choices),
        field("name", "Run name", "text", "", help_="empty: <dataset>_<model>_d<dim>_<date-time>"),
        field("steps", "Steps", "text", env("STEPS"), check=is_int),
        field("save", "Save every", "text", env("SAVE_EVERY"), check=is_int, help_="checkpoint + resume state every N steps"),
        field("lr", "Learning rate", "text", env("LR"), check=is_lr),
        field("dim", "LoRA dim", "text", env("DIM"), check=is_int),
        field("alpha", "LoRA alpha", "text", env("ALPHA"), check=is_int),
        field("res", "Resolution", "text", env("RESOLUTION"), check=is_int),
        field("buckets", "Buckets", "text", "%s:%s" % (env("MIN_BUCKET"), env("MAX_BUCKET")), check=is_buckets,
              help_="MIN:MAX bucket resolution"),
        field("clip", "Clip skip", "choice", 0, ["1 (SD 1.5 default)", "2 (most anime models)"]),
        field("prec", "Precision", "choice", 0, ["fp16 (mixed precision)", "fp32"],
              help_="fp16: faster and 2.2 GB less VRAM; fp32: the most thoroughly checked path"),
        field("te", "Text encoder", "choice", 0, ["frozen (UNet only)", "trained"],
              help_="training it helps prompts respond to the trigger; it overfits quickly"),
        field("te_lr", "  TE rate", "text", env("TE_LR", "5e-5"), check=is_lr),
        field("vram", "VRAM limit", "choice", 0, ["auto", "reserve MiB", "cap MiB", "no cap"],
              help_="auto: leave what the card's other users hold + HIP runtime"),
        field("vram_mib", "  MiB", "text", "", check=is_mib),
    ]
    actions = list(actions)
    sel = 0

    def value_text(f):
        if f["kind"] in ("pick", "choice"):
            return f["options"][f["value"]] if f["options"] else "(none found)"
        return f["value"] or ("(automatic)" if f["key"] == "name" else "")

    def visible():
        vram = fields[[f["key"] for f in fields].index("vram")]["value"]
        te = fields[[f["key"] for f in fields].index("te")]["value"]
        return [f for f in fields if (f["key"] != "vram_mib" or vram in (1, 2)) and (f["key"] != "te_lr" or te == 1)]

    def describe_set(i):
        s = sets[i]
        warn = "" if s["captions"] == s["images"] else "  - %d without captions" % (s["images"] - s["captions"])
        return "%d images, %d captioned in %s%s" % (s["images"], s["captions"], ", ".join(s["concepts"]) or "no <repeats>_<trigger>/ folders", warn)

    while True:
        items = visible()
        entries = items + actions
        sel = min(sel, len(entries) - 1)
        h, w = m.scr.getmaxyx()
        m.frame("new run", "Up/Down move   Left/Right change   Enter edit/choose   Esc back   q quit")
        for i, f in enumerate(items):
            bad = f["check"] and f["value"] and not f["check"](f["value"])
            m.put(2 + i, 2, ("> " if i == sel else "  ") + "%-14s" % f["label"], curses.A_REVERSE if i == sel else 0)
            m.put(2 + i, 19, value_text(f), m.attr(3) if bad else 0)
        for j, a in enumerate(actions):
            i = len(items) + j
            m.put(3 + i, 2, ("> " if i == sel else "  ") + "[ %s ]" % a, curses.A_REVERSE if i == sel else curses.A_BOLD)
        cur = entries[sel]
        hint = describe_set(fields[0]["value"]) if cur is fields[0] and sets else (cur.get("help", "") if isinstance(cur, dict) else "")
        m.put(h - 3, 2, hint, m.attr(2))
        m.put(h - 2, 2, "gfx803-train " + " ".join(shlex.quote(a) for a in build_args(fields, sets, models, vaes, [])), curses.A_DIM)

        key = m.scr.getch()
        if key in (curses.KEY_UP, ord("k")):
            sel = (sel - 1) % len(entries)
        elif key in (curses.KEY_DOWN, ord("j"), 9):
            sel = (sel + 1) % len(entries)
        elif key == 27:
            return None
        elif key == ord("q"):
            raise Quit()
        elif isinstance(cur, dict) and cur["kind"] in ("pick", "choice") and key in (curses.KEY_LEFT, curses.KEY_RIGHT):
            if cur["options"]:
                cur["value"] = (cur["value"] + (1 if key == curses.KEY_RIGHT else -1)) % len(cur["options"])
        elif key in (curses.KEY_ENTER, 10, 13):
            if isinstance(cur, dict):
                if cur["kind"] == "pick" and cur["options"]:
                    i = m.pick("choose " + cur["label"].lower(), cur["options"], cur["value"],
                               describe=describe_set if cur is fields[0] else None)
                    if i is not None:
                        cur["value"] = i
                elif cur["kind"] == "choice":
                    cur["value"] = (cur["value"] + 1) % len(cur["options"])
                else:
                    new = m.edit(2 + sel, 19, cur["value"], max(w - 22, 10))
                    if new is not None:
                        cur["value"] = new
            else:
                action = actions[sel - len(items)]
                if action == "Cancel":
                    return None
                bad = [f["label"].strip() for f in items if f["check"] and not f["check"](f["value"])
                       and not (f["key"] == "name")]
                if not sets:
                    bad.append("Dataset (none found)")
                if not models:
                    bad.append("Base model (none found)")
                if bad:
                    m.put(h - 3, 2, " " * (w - 3))
                    m.put(h - 3, 2, "Fix first: " + ", ".join(bad), m.attr(3, curses.A_BOLD))
                    m.scr.getch()
                    continue
                return build_args(fields, sets, models, vaes, ["--dry-run"] if action.startswith("Dry") else [])


def args_for(c, extra=()):
    """gfx803-train arguments for a dict of choices, leaving out settings
    that equal the wrapper's defaults so the command stays readable.

    Keys: data, model (path), vae (None = default, "none", or a path), name,
    steps, save, lr, dim, alpha, res, buckets ("MIN:MAX"), clip_skip (1/2),
    vram ("auto", "none", ("reserve", MiB) or ("cap", MiB)), te (bool), te_lr,
    precision ("fp16" or "fp32").
    """
    args = ["--data", c["data"]]
    if c.get("model"):
        args += ["--model", c["model"]]
    if c.get("vae"):
        args += ["--vae", c["vae"]]
    if c.get("name"):
        args += ["--name", c["name"]]
    for key, flag, default in (("steps", "--steps", "STEPS"), ("save", "--save-every", "SAVE_EVERY"),
                               ("lr", "--lr", "LR"), ("dim", "--dim", "DIM"), ("alpha", "--alpha", "ALPHA"),
                               ("res", "--resolution", "RESOLUTION")):
        value = str(c.get(key) or "")
        if value and value != env(default):
            args += [flag, value]
    buckets = c.get("buckets") or ""
    if buckets and buckets != "%s:%s" % (env("MIN_BUCKET"), env("MAX_BUCKET")):
        args += ["--buckets", buckets]
    if c.get("precision") == "fp32":
        args += ["--precision", "fp32"]
    if c.get("te"):
        args += ["--train-text-encoder"]
        if c.get("te_lr") and str(c["te_lr"]) != env("TE_LR"):
            args += ["--te-lr", str(c["te_lr"])]
    vram = c.get("vram", "auto")
    if vram == "none":
        args += ["--vram-cap", "none"]
    elif isinstance(vram, tuple):
        args += ["--vram-reserve" if vram[0] == "reserve" else "--vram-cap", str(vram[1])]
    args += ["--resume", "never"] + list(extra)
    if c.get("clip_skip") == 2:
        args += ["--", "--clip_skip", "2"]
    return args


def build_args(fields, sets, models, vaes, extra):
    f = {x["key"]: x for x in fields}
    if not sets:
        return []
    vae_i, vram_i, mib = f["vae"]["value"], f["vram"]["value"], f["vram_mib"]["value"]
    c = {
        "data": sets[f["data"]["value"]]["name"],
        "model": os.path.join(env("CKPT_DIR"), models[f["model"]["value"]]) if models else None,
        "vae": None if vae_i == 0 else ("none" if vae_i == 1 else os.path.join(env("VAE_DIR"), vaes[vae_i - 2])),
        "name": f["name"]["value"],
        "steps": f["steps"]["value"], "save": f["save"]["value"], "lr": f["lr"]["value"],
        "dim": f["dim"]["value"], "alpha": f["alpha"]["value"], "res": f["res"]["value"],
        "buckets": f["buckets"]["value"],
        "clip_skip": 2 if f["clip"]["value"] == 1 else 1,
        "precision": "fp32" if f["prec"]["value"] == 1 else "fp16",
        "te": f["te"]["value"] == 1, "te_lr": f["te_lr"]["value"],
        "vram": "auto" if vram_i == 0 or (vram_i in (1, 2) and not mib)
                else "none" if vram_i == 3 else (("reserve" if vram_i == 1 else "cap"), mib),
    }
    return args_for(c, extra)


def main_menu(m):
    sets = datasets(env("DATA_ROOT"))
    models = model_files(env("CKPT_DIR"), MODEL_EXT)
    vaes = model_files(env("VAE_DIR"), VAE_EXT)
    while True:
        runs = unfinished_runs(env("RUNS_DIR"), env("OUT_ROOT"))
        choices = ["New training run"]
        if runs:
            choices.append("Resume an unfinished run (%d)" % len(runs))
        choices.append("Quit")
        i = m.pick("what do you want to do?", choices)
        if i is None or choices[i] == "Quit":
            return None
        if choices[i] == "New training run":
            args = new_run_form(m, sets, models, vaes)
            if args is not None:
                return args
        else:
            labels = ["%-50s %-12s step %s/%s%s" % (r["name"], r["state"], r["done"], r["total"],
                                                    "  (may still be training)" if r["state"] == "running" else "")
                      for r in runs]
            j = m.pick("resume which run?", labels)
            if j is not None:
                return ["--resume-run", runs[j]["name"]]


def main():
    out = sys.argv[1]
    try:
        args = curses.wrapper(lambda scr: main_menu(Menu(scr)))
    except Quit:
        args = None
    except KeyboardInterrupt:
        args = None
    if args is None:
        return 1
    with open(out, "wb") as f:
        f.write(b"\0".join(a.encode() for a in args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
