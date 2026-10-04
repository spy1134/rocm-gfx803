"""Dataset tools for gfx803-app: captions, tags and images of sd-scripts
datasets (<set>/<repeats>_<trigger>/ folders of images and .txt captions).

Every change backs up the files it touches first, under
$WORKSPACE/dataset_backups/<set>/<time>-<action>/ with a manifest, and
undo_last() restores the newest one. Backups live outside training_data/
because sd-scripts reads every subfolder of a dataset and warns about any
that does not start with a repeat count.
"""
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(WORKSPACE, "training_data"))
BACKUP_ROOT = os.path.join(WORKSPACE, "dataset_backups")
TAGGER_MODELS = os.path.join(WORKSPACE, "models", "wd14")
SD_SCRIPTS = os.environ.get("SD_SCRIPTS", "/opt/sd-scripts")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
TAGGER_REPOS = ["SmilingWolf/wd-swinv2-tagger-v3", "SmilingWolf/wd-eva02-large-tagger-v3",
                "SmilingWolf/wd-vit-tagger-v3", "SmilingWolf/wd-convnext-tagger-v3"]


# ------------------------------------------------------------------ reading
def concept_dirs(set_path):
    return sorted(os.path.join(set_path, d) for d in os.listdir(set_path)
                  if re.match(r"^\d+_", d) and os.path.isdir(os.path.join(set_path, d)))


def images(set_path):
    """[(image path, caption path)] for every image in the set's concept folders."""
    out = []
    for d in concept_dirs(set_path):
        for f in sorted(os.listdir(d)):
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMAGE_EXT:
                out.append((os.path.join(d, f), os.path.join(d, stem + ".txt")))
    return out


def read_tags(caption):
    try:
        with open(caption) as f:
            return [t.strip() for t in f.read().replace("\n", ",").split(",") if t.strip()]
    except OSError:
        return None


def write_tags(caption, tags):
    with open(caption, "w") as f:
        f.write(", ".join(tags) + "\n")


def summary(set_path):
    """Counts and tag statistics for a set."""
    items = images(set_path)
    freq = collections.Counter()
    first = collections.Counter()
    captioned = 0
    for _img, cap in items:
        tags = read_tags(cap)
        if tags is None:
            continue
        captioned += 1
        freq.update(set(tags))
        if tags:
            first[tags[0]] += 1
    trigger = None
    if captioned and first:
        tag, n = first.most_common(1)[0]
        if n == captioned:
            trigger = tag
    return {"images": len(items), "captioned": captioned, "uncaptioned": len(items) - captioned,
            "freq": freq, "trigger": trigger,
            "concepts": [os.path.basename(d) for d in concept_dirs(set_path)]}


def suggested_trigger(set_path):
    """The trigger from the first concept folder's name (10_mychar -> mychar)."""
    dirs = concept_dirs(set_path)
    return os.path.basename(dirs[0]).split("_", 1)[1] if dirs else ""


def latent_caches(img):
    """sd-scripts' cached latents for an image (<stem>_<W>x<H>_sd.npz). Their
    names encode only the resolution, so after the image's pixels change, the
    old cache would keep being trained on unless it is removed."""
    d, stem = os.path.split(os.path.splitext(img)[0])
    pat = re.compile(re.escape(stem) + r"_\d+x\d+_sd\.npz$")
    try:
        return [os.path.join(d, f) for f in os.listdir(d) if pat.match(f)]
    except OSError:
        return []


def image_info(path):
    """(width, height, mode, has transparency) of one image, or None."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            w, h, mode = im.width, im.height, im.mode
            if im.mode == "P" and "transparency" in im.info:
                im = im.convert("RGBA")
            transparent = im.mode in ("RGBA", "LA", "PA") and im.getchannel("A").getextrema()[0] < 255
            return w, h, mode, transparent
    except OSError:
        return None


def transparent_images(set_path):
    """Images with at least one pixel that is not fully opaque. sd-scripts
    converts them to RGB, which turns transparent areas into whatever color
    their pixels happen to hold, often black, and the LoRA learns it."""
    from PIL import Image
    out = []
    for img, _cap in images(set_path):
        try:
            with Image.open(img) as im:
                if im.mode == "P" and "transparency" in im.info:
                    im = im.convert("RGBA")
                if im.mode in ("RGBA", "LA", "PA"):
                    alpha = im.getchannel("A")
                    if alpha.getextrema()[0] < 255:
                        out.append(img)
        except OSError:
            continue
    return out


# ------------------------------------------------------------------ backups
class Backup:
    """Copies files into a new backup folder before they are changed."""

    def __init__(self, set_path, action):
        self.set_path = set_path
        name = os.path.basename(set_path.rstrip("/"))
        self.dir = os.path.join(BACKUP_ROOT, name, time.strftime("%Y%m%d-%H%M%S-") + action)
        n = 1
        while os.path.exists(self.dir):
            self.dir = f"{self.dir.rsplit('~', 1)[0]}~{n}"
            n += 1
        self.manifest = {"action": action, "time": time.time(), "set": set_path, "saved": [], "created": []}

    def rel(self, path):
        return os.path.relpath(path, self.set_path)

    def save(self, path):
        """Records path before it changes: a copy if it exists, else a note
        that it is new, so undo deletes it."""
        rel = self.rel(path)
        if rel in self.manifest["saved"] or rel in self.manifest["created"]:
            return
        if os.path.exists(path):
            dst = os.path.join(self.dir, "files", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(path, dst)
            self.manifest["saved"].append(rel)
        else:
            self.manifest["created"].append(rel)

    def commit(self):
        """Writes the manifest; safe to call again after more save() calls,
        so a long editing session stays undoable even if the app dies."""
        if not (self.manifest["saved"] or self.manifest["created"]):
            return None
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump(self.manifest, f, indent=1)
        return self.dir


def backups(set_path):
    """Backup folders of a set, newest first, as (dir, manifest)."""
    root = os.path.join(BACKUP_ROOT, os.path.basename(set_path.rstrip("/")))
    out = []
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for n in names:
        try:
            with open(os.path.join(root, n, "manifest.json")) as f:
                out.append((os.path.join(root, n), json.load(f)))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda b: b[1]["time"], reverse=True)
    return out


def undo_last(set_path):
    """Restores the newest backup and removes it; returns its manifest or None."""
    bks = backups(set_path)
    if not bks:
        return None
    d, m = bks[0]
    for rel in m["saved"]:
        shutil.copy2(os.path.join(d, "files", rel), os.path.join(set_path, rel))
    for rel in m["created"]:
        try:
            os.remove(os.path.join(set_path, rel))
        except FileNotFoundError:
            pass
    shutil.rmtree(d)
    return m


# ------------------------------------------------------------------ changes
def set_trigger(set_path, trigger):
    """Puts trigger first in every caption (creating captions that are
    missing) and drops other copies of it. Returns the number changed."""
    trigger = trigger.strip()
    b = Backup(set_path, "trigger")
    changed = 0
    for _img, cap in images(set_path):
        tags = read_tags(cap) or []
        new = [trigger] + [t for t in tags if t != trigger]
        if new != tags:
            b.save(cap)
            write_tags(cap, new)
            changed += 1
    b.commit()
    return changed


def remove_tags(set_path, tags, keep=()):
    """Removes the given tags from every caption. Returns (captions changed,
    tag occurrences removed)."""
    drop = {t.strip() for t in tags if t.strip()} - set(keep)
    b = Backup(set_path, "remove-tags")
    changed = removed = 0
    for _img, cap in images(set_path):
        old = read_tags(cap)
        if old is None:
            continue
        new = [t for t in old if t not in drop]
        if len(new) != len(old):
            b.save(cap)
            write_tags(cap, new)
            changed += 1
            removed += len(old) - len(new)
    b.commit()
    return changed, removed


def flatten(set_path, paths, color=(255, 255, 255)):
    """Composites transparent images onto a solid background, in place and in
    their own format. Returns the number of images written."""
    from PIL import Image
    b = Backup(set_path, "flatten")
    n = 0
    for p in paths:
        with Image.open(p) as im:
            fmt = im.format
            rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, color + (255,))
        out = Image.alpha_composite(bg, rgba).convert("RGB")
        b.save(p)
        save_kwargs = {"quality": 95} if fmt in ("JPEG", "WEBP") else {}
        out.save(p, format=fmt, **save_kwargs)
        for cache in latent_caches(p):
            b.save(cache)
            os.remove(cache)
        n += 1
    b.commit()
    return n


def delete_image(set_path, img, cap):
    """Removes an image with its caption and latent caches. They move into a
    backup rather than being erased, so undo_last() brings them back."""
    b = Backup(set_path, "delete-image")
    for p in [img, cap] + latent_caches(img):
        if os.path.exists(p):
            b.save(p)
            os.remove(p)
    b.commit()


# ------------------------------------------------------------------ tagging
TAG_SCRIPT = r'''
import runpy, sys, onnxruntime
# The ROCm execution provider is not validated on gfx803; tagging a dataset
# takes a minute or two on the CPU.
onnxruntime.get_available_providers = lambda: ["CPUExecutionProvider"]
sys.argv = ["tag_images_by_wd14_tagger.py"] + sys.argv[1:]
sys.path.insert(0, SD_SCRIPTS)
runpy.run_path(SD_SCRIPTS + "/finetune/tag_images_by_wd14_tagger.py", run_name="__main__")
'''


def start_tagging(set_path, repo, general=0.35, character=0.85, mode="untagged", trigger="", log_path=None):
    """Starts WD14 tagging in the background; returns (Popen, finish), where
    finish() must be called once the process has exited to move captions into
    place, apply the trigger and record the backup.

    mode: "untagged" (only images without a caption), "overwrite" (retag all)
    or "append" (add new tags to existing captions).
    """
    items = images(set_path)
    if mode == "untagged":
        items = [(i, c) for i, c in items if not os.path.exists(c)]
    b = Backup(set_path, "tag")
    for _img, cap in items:
        b.save(cap)
    work = tempfile.mkdtemp(prefix="gfx803-tag-", dir=WORKSPACE)
    links = []
    for img, cap in items:
        rel = os.path.relpath(img, set_path)
        link = os.path.join(work, rel)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(img, link)
        if mode == "append" and os.path.exists(cap):
            shutil.copy2(cap, os.path.splitext(link)[0] + ".txt")
        links.append((link, cap))
    os.makedirs(TAGGER_MODELS, exist_ok=True)
    args = [sys.executable, "-c", TAG_SCRIPT.replace("SD_SCRIPTS", repr(SD_SCRIPTS)),
            "--onnx", "--repo_id", repo, "--model_dir", TAGGER_MODELS, "--batch_size", "1",
            "--general_threshold", str(general), "--character_threshold", str(character),
            "--remove_underscore", "--caption_extension", ".txt", "--recursive"]
    if mode == "append":
        args.append("--append_tags")
    args.append(work)
    log = open(log_path or os.devnull, "ab")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} tagging {len(items)} images of "
              f"{set_path} with {repo} ({mode})\n".encode())
    log.flush()
    proc = subprocess.Popen(args, cwd=SD_SCRIPTS, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)

    def finish():
        moved = 0
        for link, cap in links:
            produced = os.path.splitext(link)[0] + ".txt"
            if os.path.exists(produced):
                shutil.move(produced, cap)
                moved += 1
        shutil.rmtree(work, ignore_errors=True)
        if trigger.strip():
            for _link, cap in links:
                tags = read_tags(cap)
                if tags is not None:
                    write_tags(cap, [trigger.strip()] + [t for t in tags if t != trigger.strip()])
        b.commit()
        return moved, len(links)

    return proc, finish
