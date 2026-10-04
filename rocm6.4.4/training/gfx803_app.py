#!/usr/bin/env python3
"""gfx803-app: terminal UI for LoRA training and the ComfyUI server.

Watches the GPU and the system, edits the training queue, starts and stops
ComfyUI, and copies finished LoRAs to ComfyUI. The work itself is done by
gfx803-service (the container's main process); this app only reads status
files and writes requests to the shared queue file, so quitting it stops
nothing.
"""
import collections
import contextlib
import math
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import time

import psutil

# textual-image asks the terminal which graphics protocols it supports when it
# is imported, which only works before Textual takes over the terminal, so it
# is imported first. GFX803_IMAGE_MODE=sixel|tgp|halfcell|unicode overrides
# the detection for terminals that answer wrongly.
import io as _io

try:
    # A terminal that ignores the queries costs a 2 s timeout, which the
    # library reports on stderr before falling back; keep that off the screen.
    _err, sys.stderr = sys.stderr, _io.StringIO()
    try:
        import textual_image.widget as timg
    finally:
        sys.stderr = _err
    _IMAGE_MODES = {"sixel": timg.SixelImage, "tgp": timg.TGPImage, "halfcell": timg.HalfcellImage,
                    "unicode": timg.UnicodeImage}
    PreviewImage = _IMAGE_MODES.get(os.environ.get("GFX803_IMAGE_MODE", "").lower(), timg.Image)
except Exception as _e:  # noqa: BLE001 - previews are optional; the rest of the app must start
    from textual.widgets import Static as _Static

    class PreviewImage(_Static):
        """Stand-in when this terminal's image support could not be set up."""

        _reason = f"Image preview unavailable in this terminal ({type(_e).__name__})."

        def __init__(self, *args, **kwargs):
            super().__init__(self._reason, *args, **kwargs)

        @property
        def image(self):
            return None

        @image.setter
        def image(self, _value):
            pass
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label, Log, ProgressBar, Select,
                             SelectionList, Sparkline, Static, Switch, TabbedContent, TabPane, TextArea)
from textual_plotext import PlotextPlot

import gfx803_dataset as ds
import gfx803_generate as gen
import gfx803_queue as gq
import gfx803_status as st
import gfx803_train_menu as menu

HISTORY = 600  # one sample a second: ten minutes of graphs
DEFAULT_S_PER_STEP = 4.3
LORAS_ROOT = "/opt/ComfyUI/models/loras"
# Everything gfx803-train makes is an SD 1.5 LoRA, so that folder is the default.
DEFAULT_LORA_SUBDIR = "SD15"
SERVICE_STALE_S = 5

STATE_STYLE = {gq.RUNNING: "bold cyan", gq.PENDING: "", gq.DONE: "green", gq.FAILED: "bold red",
               gq.STOPPED: "yellow", gq.SKIPPED: "dim"}


def fmt_duration(seconds):
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    return f"{h} h {rem // 60:02d} min" if h else f"{rem // 60} min {rem % 60:02d} s"


def fmt_eta(seconds):
    if seconds is None:
        return "?"
    return fmt_duration(seconds) + time.strftime(" (done %H:%M)", time.localtime(time.time() + seconds))


def comfy_urls():
    """Where ComfyUI can be reached, following gfx803-service's listen address."""
    listen = os.environ.get("GFX803_COMFYUI_LISTEN") or "127.0.0.1"
    if listen in ("127.0.0.1", "localhost", "::1"):
        return ["http://localhost:8188"]
    if listen in ("0.0.0.0", "::"):
        return [f"http://{a}:8188" for a in lan_addresses()] or ["http://localhost:8188"]
    return [f"http://{listen}:8188"]


def lan_addresses():
    addrs = []
    for name, entries in psutil.net_if_addrs().items():
        if name.startswith(("lo", "docker", "br-", "veth")):
            continue
        for e in entries:
            if e.family == socket.AF_INET and not e.address.startswith("169.254."):
                addrs.append(e.address)
    return addrs


def load_defaults():
    """Folders and defaults from gfx803-train itself, so forms offer what a
    plain gfx803-train run would use."""
    try:
        out = subprocess.run(gq.TRAIN_CMD + ["--print-defaults"], capture_output=True, text=True, timeout=30,
                             env=dict(os.environ, GFX803_TRAIN_NO_MENU="1")).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    for line in out.splitlines():
        k, sep, v = line.partition("=")
        if sep and k.isupper():
            os.environ.setdefault(k, v)


# ------------------------------------------------------------------ dialogs
class Confirm(ModalScreen[bool]):
    def __init__(self, message, ok="Yes", danger=False):
        super().__init__()
        self.message, self.ok, self.danger = message, ok, danger

    def compose(self):
        with Vertical(classes="dialog"):
            yield Label(self.message, classes="dialog-text")
            with Horizontal(classes="buttons"):
                yield Button(self.ok, variant="error" if self.danger else "primary", id="ok")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed)
    def done(self, event):
        self.dismiss(event.button.id == "ok")

    def key_escape(self):
        self.dismiss(False)


class LogView(ModalScreen):
    """Tail of a log file, refreshed while open."""

    BINDINGS = [Binding("escape,q", "app.pop_screen", "Close")]

    def __init__(self, title, path):
        super().__init__()
        self.title_text, self.path, self.log_size = title, path, -1

    def compose(self):
        with Vertical(classes="dialog wide"):
            yield Label(f"[b]{self.title_text}[/b]  [dim]{self.path}[/dim]")
            yield Log(id="log", highlight=False, max_lines=5000)
            with Horizontal(classes="buttons"):
                yield Button("Close", id="close")

    def on_mount(self):
        self.reload()
        self.set_interval(2, self.reload)

    def reload(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            size = 0
        if size == self.log_size:
            return
        self.log_size = size
        lines = [l for l in st.tail(self.path, 300000).split("\n") if l.strip()]
        log = self.query_one(Log)
        log.clear()
        log.write_lines(lines[-3000:])

    @on(Button.Pressed, "#close")
    def close(self):
        self.app.pop_screen()


class PickCheckpoint(ModalScreen):
    """Copy one of a run's checkpoints into ComfyUI's LoRA folders, with a
    chosen subfolder and file name."""

    def __init__(self, run):
        super().__init__()
        self.ckpt_run = run
        env = gq.read_kv(os.path.join(gq.RUNS_DIR, run, "run.env"))
        data = os.path.basename(env.get("DATA_DIR", "").rstrip("/").strip("'\"")) or run
        self.short = re.sub(r"[^A-Za-z0-9._-]+", "_", data).strip("_") or "lora"

    def compose(self):
        cks = st.checkpoints(self.ckpt_run)
        options = [(f"final ({os.path.basename(p)})" if s is None else f"step {s}", p) for s, p in reversed(cks)]
        try:
            subdirs = sorted(d for d in os.listdir(LORAS_ROOT) if os.path.isdir(os.path.join(LORAS_ROOT, d)))
        except OSError:
            subdirs = []
        if DEFAULT_LORA_SUBDIR not in subdirs:
            subdirs.insert(0, DEFAULT_LORA_SUBDIR)
        folder_opts = [(f"loras/{d}/", d) for d in subdirs] + [("loras/ (top level)", ""), ("new folder...", "__new__")]
        with Vertical(classes="dialog"):
            yield Label(f"[b]Copy a LoRA to ComfyUI[/b]\nfrom {self.ckpt_run}", classes="dialog-text")
            if not options:
                yield Label("[red]This run has no checkpoints.[/red]")
            else:
                with Grid(classes="fields copy-fields"):
                    yield Label("Checkpoint")
                    yield Select(options, value=options[0][1], allow_blank=False, id="ck")
                    yield Label("Folder")
                    yield Select(folder_opts, value=DEFAULT_LORA_SUBDIR, allow_blank=False, id="folder")
                    yield Label("New folder")
                    yield Input(placeholder="e.g. SDXL", id="newfolder", disabled=True)
                    yield Label("File name")
                    yield Input(id="fname")
                yield Static("", id="dest", classes="preview")
            with Horizontal(classes="buttons"):
                yield Button("Copy", variant="primary", id="copy", disabled=not options)
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        if self.query("#ck"):
            self.query_one("#fname", Input).value = self.default_name(self.query_one("#ck", Select).value)
            self.update_dest()

    def default_name(self, src):
        m = re.search(r"-step0*(\d+)\.safetensors$", src)
        return f"{self.short}-{'step' + m.group(1) if m else 'final'}.safetensors"

    def destination(self):
        """(folder, path) for the current choices, or (None, reason)."""
        folder = self.query_one("#folder", Select).value
        if folder == "__new__":
            folder = self.query_one("#newfolder", Input).value.strip().strip("/")
            if not re.fullmatch(r"[A-Za-z0-9._ -]+", folder or "") or folder in (".", ".."):
                return None, "enter a folder name (letters, digits, . _ - and spaces)"
        name = self.query_one("#fname", Input).value.strip()
        if not name.endswith(".safetensors"):
            name += ".safetensors"
        if "/" in name or name.startswith(".") or name == ".safetensors":
            return None, "enter a file name without slashes"
        d = os.path.join(LORAS_ROOT, folder)
        return d, os.path.join(d, name)

    @on(Select.Changed, "#ck")
    def checkpoint_changed(self, event):
        fname = self.query_one("#fname", Input)
        # Follow the checkpoint unless the user typed a name of their own.
        if not fname.value or re.fullmatch(re.escape(self.short) + r"-(step\d+|final)\.safetensors", fname.value):
            fname.value = self.default_name(event.value)
        self.update_dest()

    @on(Select.Changed, "#folder")
    @on(Input.Changed)
    def update_dest(self, _event=None):
        if not self.query("#folder"):
            return
        self.query_one("#newfolder", Input).disabled = self.query_one("#folder", Select).value != "__new__"
        d, dst = self.destination()
        if d is None:
            self.query_one("#dest", Static).update(f"[red]{dst}[/red]")
            self.query_one("#copy", Button).disabled = True
            return
        exists = os.path.exists(dst)
        host = os.path.relpath(dst, "/opt")
        self.query_one("#dest", Static).update(
            f"copies to [b]{host}[/b]" + ("   [yellow]exists: will ask before replacing[/yellow]" if exists else ""))
        self.query_one("#copy", Button).disabled = False

    @on(Button.Pressed)
    def done(self, event):
        if event.button.id == "copy":
            self.copy()
        else:
            self.dismiss(None)

    @work
    async def copy(self):
        src = self.query_one("#ck", Select).value
        d, dst = self.destination()
        if d is None:
            return
        if os.path.exists(dst) and not await self.app.push_screen_wait(
                Confirm(f"{os.path.relpath(dst, '/opt')} already exists. Replace it?", ok="Replace", danger=True)):
            return
        try:
            os.makedirs(d, exist_ok=True)
            tmp = dst + ".part"
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            self.app.notify(f"Copied to {os.path.relpath(dst, '/opt')}\nRefresh ComfyUI's LoRA list to see it.",
                            title="LoRA copied")
        except OSError as e:
            self.app.notify(str(e), title="Copy failed", severity="error")
            return
        self.dismiss(None)

    def key_escape(self):
        self.dismiss(None)


class TagTable(DataTable):
    BINDINGS = [Binding("d,D,delete", "app.remove_tag", "Remove tag")]


class TagDialog(ModalScreen):
    """Options for WD14 tagging of one dataset."""

    def __init__(self, set_name, info, trigger):
        super().__init__()
        self.set_name, self.info, self.trigger = set_name, info, trigger

    def compose(self):
        i = self.info
        modes = [(f"only images without a caption ({i['uncaptioned']})", "untagged"),
                 (f"all images, replacing captions ({i['images']})", "overwrite"),
                 ("all images, adding new tags to existing captions", "append")]
        with Vertical(classes="dialog form-small"):
            yield Label(f"[b]Tag {self.set_name} with WD14[/b]\nRuns on the CPU; about 1-2 s per image. The model "
                        f"downloads on first use into workspace/models/wd14.", classes="dialog-text")
            with Grid(classes="fields copy-fields"):
                yield Label("Images")
                yield Select(modes, value="untagged" if i["uncaptioned"] else "overwrite", allow_blank=False, id="mode")
                yield Label("Model")
                yield Select([(r.split("/")[1], r) for r in ds.TAGGER_REPOS], value=ds.TAGGER_REPOS[0],
                             allow_blank=False, id="repo")
                yield Label("General")
                yield Input("0.35", id="general")
                yield Label("Character")
                yield Input("0.85", id="character")
                yield Label("Trigger")
                yield Input(self.trigger, placeholder="put first in every new caption (empty: none)", id="trigger")
            yield Label("[dim]Thresholds: how confident the tagger must be to write a tag. Lower = more tags.[/dim]")
            with Horizontal(classes="buttons"):
                yield Button("Start tagging", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed)
    def done(self, event):
        if event.button.id != "ok":
            self.dismiss(None)
            return
        try:
            g, c = float(self.query_one("#general", Input).value), float(self.query_one("#character", Input).value)
        except ValueError:
            self.app.notify("Thresholds must be numbers between 0 and 1.", severity="error")
            return
        self.dismiss({"mode": self.query_one("#mode", Select).value, "repo": self.query_one("#repo", Select).value,
                      "general": g, "character": c, "trigger": self.query_one("#trigger", Input).value.strip()})

    def key_escape(self):
        self.dismiss(None)


class TextDialog(ModalScreen):
    """One text field with an explanation; returns the text or None."""

    def __init__(self, title, text, value="", placeholder="", ok="OK"):
        super().__init__()
        self.title_text, self.text, self.value, self.placeholder, self.ok = title, text, value, placeholder, ok

    def compose(self):
        with Vertical(classes="dialog form-small"):
            yield Label(f"[b]{self.title_text}[/b]", classes="dialog-title")
            yield Label(self.text, classes="dialog-text")
            yield Input(self.value, placeholder=self.placeholder, id="text")
            with Horizontal(classes="buttons"):
                yield Button(self.ok, variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    @on(Input.Submitted)
    def submitted(self):
        self.dismiss(self.query_one("#text", Input).value)

    @on(Button.Pressed)
    def done(self, event):
        self.dismiss(self.query_one("#text", Input).value if event.button.id == "ok" else None)

    def key_escape(self):
        self.dismiss(None)


def filter_images(items, query):
    """Images whose caption matches a filter: comma-separated tags that must
    all be present, "-tag" for tags that must be absent, and "uncaptioned"."""
    want, avoid, uncaptioned = [], [], False
    for part in (p.strip() for p in query.split(",")):
        if not part:
            continue
        if part == "uncaptioned":
            uncaptioned = True
        elif part.startswith("-"):
            avoid.append(part[1:].strip())
        else:
            want.append(part)
    out = []
    for img, cap in items:
        tags = ds.read_tags(cap)
        if uncaptioned and tags is not None:
            continue
        tags = tags or []
        if all(t in tags for t in want) and not any(t in tags for t in avoid):
            out.append((img, cap))
    return out


class ImageFull(Screen):
    """One image filling the terminal, with zoom and pan for close inspection
    and its caption underneath."""

    BINDINGS = [
        Binding("escape,q,enter", "close", "Back"),
        Binding("plus,equals_sign", "zoom(1)", "Zoom in"),
        Binding("minus", "zoom(-1)", "Zoom out"),
        Binding("0", "zoom(0)", "Fit"),
        Binding("left", "move(-1, 0)", "Left/prev", show=False),
        Binding("right", "move(1, 0)", "Right/next", show=False),
        Binding("up", "move(0, -1)", show=False),
        Binding("down", "move(0, 1)", show=False),
        Binding("pagedown,n,space", "step(1)", "Next image"),
        Binding("pageup,p", "step(-1)", "Previous image"),
    ]
    ZOOMS = [1, 1.5, 2, 3, 4, 6, 8]

    def __init__(self, items, index, on_index=None, describe=None):
        super().__init__()
        self.items, self.index, self.on_index = items, index, on_index
        self.describe = describe
        self.zoom_i, self.cx, self.cy = 0, 0.5, 0.5

    def compose(self):
        yield Label("", id="full-title")
        with Container(id="full-box"):
            yield PreviewImage(id="full-img")
        yield Label("", id="full-caption")
        yield Footer()

    def on_mount(self):
        self.show()

    def show(self):
        from PIL import Image as PILImage
        img, cap = self.items[self.index]
        zoom = self.ZOOMS[self.zoom_i]
        try:
            with PILImage.open(img) as im:
                im.load()
                w, h = im.size
                if zoom > 1:
                    cw, ch = w / zoom, h / zoom
                    x0 = min(max(self.cx * w - cw / 2, 0), w - cw)
                    y0 = min(max(self.cy * h - ch / 2, 0), h - ch)
                    self.cx, self.cy = (x0 + cw / 2) / w, (y0 + ch / 2) / h
                    view = im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch)))
                else:
                    view = im.copy()
        except OSError as e:
            self.query_one("#full-title", Label).update(f"[red]{img}: {e}[/red]")
            return
        self.query_one("#full-img").image = view
        where = f"   view centre {self.cx:.0%}, {self.cy:.0%}" if zoom > 1 else ""
        self.query_one("#full-title", Label).update(
            f"[b]{os.path.basename(img)}[/b]   {self.index + 1}/{len(self.items)}   {w}x{h}   zoom {zoom}x{where}")
        if self.describe:
            text = self.describe(img)
        else:
            tags = ds.read_tags(cap)
            text = ", ".join(tags) if tags else "[yellow]no caption[/yellow]"
        self.query_one("#full-caption", Label).update(text)

    def action_close(self):
        if self.on_index:
            self.on_index(self.index)
        self.app.pop_screen()

    def action_zoom(self, d):
        self.zoom_i = 0 if d == 0 else min(max(self.zoom_i + d, 0), len(self.ZOOMS) - 1)
        if self.zoom_i == 0:
            self.cx, self.cy = 0.5, 0.5
        self.show()

    def action_move(self, dx, dy):
        zoom = self.ZOOMS[self.zoom_i]
        if zoom == 1:
            if dx:
                self.action_step(dx)
            return
        step = 0.25 / zoom
        self.cx, self.cy = self.cx + dx * step, self.cy + dy * step
        self.show()

    def action_step(self, d):
        self.index = (self.index + d) % len(self.items)
        self.zoom_i, self.cx, self.cy = 0, 0.5, 0.5
        self.show()


class ImageBrowser(Screen):
    """A dataset's images with a preview and per-image caption editing.

    Edits are written at once; the first change to each caption in a session
    backs it up, so the dataset's Undo reverts the whole session."""

    BINDINGS = [
        Binding("escape", "close", "Back"),
        Binding("f", "fullscreen", "Full screen"),
        Binding("delete", "remove_tag", "Remove tag"),
        Binding("shift+up", "move_tag(-1)", "Tag up"),
        Binding("shift+down", "move_tag(1)", "Tag down"),
        Binding("slash", "focus_filter", "Filter"),
        Binding("D", "delete_image", "Delete image"),
    ]

    def __init__(self, set_name, filter_text=""):
        super().__init__()
        self.set_name, self.filter_text = set_name, filter_text
        self.set_path = os.path.join(ds.DATA_ROOT, set_name)
        self.all_items = ds.images(self.set_path)
        self.items = []
        self.backup = ds.Backup(self.set_path, "edit-captions")
        self.edits = 0

    def compose(self):
        yield Header()
        with Horizontal(id="ib"):
            with Vertical(id="ib-list"):
                yield Input(self.filter_text, placeholder="filter: tag, -tag, uncaptioned", id="ib-filter")
                yield DataTable(id="ib-images", cursor_type="row", zebra_stripes=True)
            with Vertical(id="ib-preview"):
                with Container(id="ib-img-box"):
                    yield PreviewImage(id="ib-img")
                yield Label("", id="ib-info")
            with Vertical(id="ib-caption"):
                yield Label("Caption", classes="section")
                yield DataTable(id="ib-tags", cursor_type="row", show_header=False)
                yield Input(placeholder="add tags (comma-separated), Enter", id="ib-add")
                with Horizontal(classes="toolbar"):
                    yield Button("Up", id="ib-up")
                    yield Button("Down", id="ib-down")
                    yield Button("Remove", id="ib-remove", variant="error")
                yield Button("Delete image", id="ib-delete", variant="error")
                yield Label("[dim]Del removes the highlighted tag, Shift+Up/Down moves it. "
                            "f or Enter on an image: full screen with zoom (+ / -) and pan (arrows).[/dim]",
                            classes="hint")
        yield Footer()

    def on_mount(self):
        self.query_one("#ib-images", DataTable).add_columns("Image", "Tags")
        self.query_one("#ib-tags", DataTable).add_column("Tag")
        self.apply_filter()
        self.query_one("#ib-images", DataTable).focus()

    def apply_filter(self):
        self.items = filter_images(self.all_items, self.query_one("#ib-filter", Input).value)
        t = self.query_one("#ib-images", DataTable)
        t.clear()
        for i, (img, cap) in enumerate(self.items):
            tags = ds.read_tags(cap)
            t.add_row(os.path.basename(img), "-" if tags is None else str(len(tags)), key=str(i))
        self.title = f"{self.set_name}: {len(self.items)} of {len(self.all_items)} images"
        self.show_current()

    def current(self):
        t = self.query_one("#ib-images", DataTable)
        if not self.items or t.cursor_row is None or t.cursor_row >= len(self.items):
            return None
        return self.items[t.cursor_row]

    def show_current(self):
        item = self.current()
        img_w = self.query_one("#ib-img")
        info = self.query_one("#ib-info", Label)
        tags_t = self.query_one("#ib-tags", DataTable)
        keep = tags_t.cursor_row
        tags_t.clear()
        if not item:
            img_w.image = None
            info.update("[dim]no images match the filter[/dim]")
            return
        img, cap = item
        img_w.image = img
        meta = ds.image_info(img)
        if meta:
            w, h, mode, transparent = meta
            info.update(f"[b]{os.path.basename(img)}[/b]   {w}x{h} {mode}"
                        + ("   [yellow]has transparency[/yellow]" if transparent else ""))
        for i, tag in enumerate(ds.read_tags(cap) or []):
            tags_t.add_row(tag, key=str(i))
        if keep is not None and tags_t.row_count:
            tags_t.move_cursor(row=min(keep, tags_t.row_count - 1))

    def write(self, tags):
        _img, cap = self.current()
        self.backup.save(cap)
        ds.write_tags(cap, tags)
        self.backup.commit()
        self.edits += 1
        row = self.query_one("#ib-images", DataTable).cursor_row
        self.query_one("#ib-images", DataTable).update_cell_at((row, 1), str(len(tags)))
        self.show_current()

    @on(DataTable.RowHighlighted, "#ib-images")
    def image_highlighted(self, _event):
        self.show_current()

    @on(DataTable.RowSelected, "#ib-images")
    def image_selected(self, _event):
        self.action_fullscreen()

    @on(Input.Changed, "#ib-filter")
    def filter_changed(self, _event):
        self.apply_filter()

    @on(Input.Submitted, "#ib-add")
    def add_tags(self, event):
        item = self.current()
        if not item or not event.value.strip():
            return
        tags = ds.read_tags(item[1]) or []
        new = [t.strip() for t in event.value.split(",") if t.strip() and t.strip() not in tags]
        if new:
            self.write(tags + new)
        event.input.value = ""

    def action_remove_tag(self):
        item = self.current()
        t = self.query_one("#ib-tags", DataTable)
        if not item or not t.row_count:
            return
        tags = ds.read_tags(item[1]) or []
        if t.cursor_row < len(tags):
            del tags[t.cursor_row]
            self.write(tags)

    def action_move_tag(self, d):
        item = self.current()
        t = self.query_one("#ib-tags", DataTable)
        if not item or not t.row_count:
            return
        tags = ds.read_tags(item[1]) or []
        i = t.cursor_row
        k = i + d
        if 0 <= i < len(tags) and 0 <= k < len(tags):
            tags[i], tags[k] = tags[k], tags[i]
            self.write(tags)
            t.move_cursor(row=k)

    def action_focus_filter(self):
        self.query_one("#ib-filter", Input).focus()

    def action_fullscreen(self):
        if not self.items:
            return
        t = self.query_one("#ib-images", DataTable)

        def back(index):
            t.move_cursor(row=index)
            self.show_current()

        self.app.push_screen(ImageFull(self.items, t.cursor_row or 0, back))

    @on(Button.Pressed)
    def buttons(self, event):
        {"ib-up": lambda: self.action_move_tag(-1), "ib-down": lambda: self.action_move_tag(1),
         "ib-remove": self.action_remove_tag, "ib-delete": self.action_delete_image}.get(event.button.id, lambda: None)()

    @work
    async def action_delete_image(self):
        item = self.current()
        if not item:
            return
        img, cap = item
        if not await self.app.push_screen_wait(Confirm(
                f"Delete {os.path.basename(img)} from {self.set_name}?\nIts caption and cached latents go with it. "
                "They are moved into the dataset's backups, and Undo last change in the Datasets tab restores them.",
                ok="Delete", danger=True)):
            return
        row = self.query_one("#ib-images", DataTable).cursor_row
        ds.delete_image(self.set_path, img, cap)
        self.edits += 1
        self.all_items = [x for x in self.all_items if x[0] != img]
        self.apply_filter()
        t = self.query_one("#ib-images", DataTable)
        if t.row_count:
            t.move_cursor(row=min(row, t.row_count - 1))
        self.app.notify(f"Deleted {os.path.basename(img)}.", title=self.set_name)

    def action_close(self):
        self.app.pop_screen()
        if self.edits:
            self.app.dataset_changed(self.set_name, f"{self.edits} change(s) to {self.set_name}; Undo last change "
                                     "reverts them, newest first.")


def describe_generation(meta):
    """One- or two-line summary of a generated image's settings."""
    if not meta:
        return "[dim]no gfx803 settings in this image (made elsewhere?)[/dim]"
    loras = ", ".join(f"{os.path.basename(l['name'])} {float(l.get('strength', 1)):g}" for l in meta.get("loras", [])
                      if l.get("name"))
    return (f"{meta.get('positive', '')}\n[dim]{os.path.basename(meta.get('checkpoint', ''))}"
            + (f" + {loras}" if loras else "") + f"  seed {meta.get('seed')}  {meta.get('steps')} steps  cfg "
            f"{meta.get('cfg')}  {meta.get('sampler')}/{meta.get('scheduler')}  clip skip {meta.get('clip_skip')}[/dim]")


class CompareDialog(ModalScreen):
    """Choose a training run's checkpoints to render with the same prompt and seed."""

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.runs = gen.training_loras()

    def compose(self):
        names = sorted(self.runs, key=lambda r: -max(os.path.getmtime(os.path.join(gen.TRAINING_OUTPUT, r)), 0))
        with Vertical(classes="dialog form-small"):
            yield Label("[b]Compare LoRA checkpoints[/b]", classes="dialog-title")
            yield Label("Renders the current prompt and settings once per chosen checkpoint, with the same seed, "
                        "in LoRA slot 1 (other LoRA slots stay as they are).", classes="dialog-text")
            if not names:
                yield Label("[red]No training runs with checkpoints in workspace/output.[/red]")
            else:
                with Grid(classes="fields copy-fields"):
                    yield Label("Training run")
                    yield Select([(n, n) for n in names], value=names[0], allow_blank=False, id="cmp-run")
                    yield Label("Strength")
                    yield Input(str(self.first_strength()), id="cmp-strength")
                    yield Label("Seed")
                    yield Input(str(self.settings.get("seed", -1)) if int(self.settings.get("seed", -1)) >= 0 else "",
                                placeholder="empty: one random seed for all", id="cmp-seed")
                    yield Label("Baseline")
                    yield Switch(value=True, id="cmp-base")
                yield Label("Checkpoints  [dim](space toggles)[/dim]")
                yield SelectionList(id="cmp-list")
            with Horizontal(classes="buttons"):
                yield Button("Queue comparison", variant="primary", id="ok", disabled=not names)
                yield Button("Cancel", id="cancel")

    def first_strength(self):
        loras = [l for l in self.settings.get("loras", []) if l.get("name")]
        return loras[0].get("strength", 1.0) if loras else 1.0

    def on_mount(self):
        if self.query("#cmp-run"):
            self.fill(self.query_one("#cmp-run", Select).value)

    @on(Select.Changed, "#cmp-run")
    def run_changed(self, event):
        self.fill(event.value)

    def fill(self, run):
        cks = self.runs.get(run, [])
        sl = self.query_one("#cmp-list", SelectionList)
        sl.clear_options()
        steps = [s for s, _ in cks if s]
        # Preselect about four evenly spaced steps plus the last, the usual
        # first look at where a LoRA starts to overfit.
        n = len(steps)
        picks = {steps[max(round((i + 1) * n / 4) - 1, 0)] for i in range(4)} if steps else set()
        for step, name in cks:
            label = f"step {step}" if step else "final"
            sl.add_option((label, name, step in picks))

    @on(Button.Pressed)
    def done(self, event):
        if event.button.id != "ok":
            self.dismiss(None)
            return
        chosen = self.query_one("#cmp-list", SelectionList).selected
        try:
            strength = float(self.query_one("#cmp-strength", Input).value)
            seed_txt = self.query_one("#cmp-seed", Input).value.strip()
            seed = int(seed_txt) if seed_txt else gen.resolve_seed(-1)
        except ValueError:
            self.app.notify("Strength and seed must be numbers.", severity="error")
            return
        if not chosen:
            self.app.notify("Choose at least one checkpoint.", severity="warning")
            return
        self.dismiss({"run": self.query_one("#cmp-run", Select).value, "names": chosen, "strength": strength,
                      "seed": seed, "baseline": self.query_one("#cmp-base", Switch).value})

    def key_escape(self):
        self.dismiss(None)


class CompareView(Screen):
    """A comparison's images side by side, in queue order."""

    BINDINGS = [Binding("escape,q", "app.pop_screen", "Back"), Binding("f,enter", "full", "Full screen")]

    def __init__(self, items):
        super().__init__()
        self.items = items  # [(path, meta)]

    def compose(self):
        yield Header()
        cols = min(len(self.items), 4)
        with Grid(id="cmp-grid") as g:
            g.styles.grid_size_columns = cols
            for path, meta in self.items:
                with Vertical(classes="cmp-cell"):
                    with Container(classes="cmp-img"):
                        yield PreviewImage(path)
                    yield Label(f"[b]{meta.get('label', '')}[/b]" if meta else os.path.basename(path), classes="cmp-label")
        yield Label(describe_generation(self.items[0][1]) if self.items else "", id="cmp-desc")
        yield Footer()

    def on_mount(self):
        first = self.items[0][1] if self.items else None
        self.title = f"Comparison: seed {first.get('seed')}" if first else "Comparison"

    def action_full(self):
        metas = dict(self.items)
        self.app.push_screen(ImageFull([(p, None) for p, _ in self.items], 0,
                                       describe=lambda p: (f"[b]{(metas.get(p) or {}).get('label', '')}[/b]  "
                                                           + describe_generation(metas.get(p)))))


class AddRun(ModalScreen):
    """Form for a new training run; adds it to the queue."""

    def compose(self):
        self.sets = menu.datasets(os.environ.get("DATA_ROOT", os.path.join(gq.WORKSPACE, "training_data")))
        ckpt_dir = os.environ.get("CKPT_DIR", "/opt/ComfyUI/models/checkpoints")
        vae_dir = os.environ.get("VAE_DIR", "/opt/ComfyUI/models/vae")
        models = menu.model_files(ckpt_dir, menu.MODEL_EXT)
        vaes = menu.model_files(vae_dir, menu.VAE_EXT)
        set_opts = [(f"{s['name']}  ({s['images']} images, {s['captions']} captioned)", s["name"]) for s in self.sets]
        model_opts = [(m, os.path.join(ckpt_dir, m)) for m in models]
        default_model = model_opts[menu.default_index(models, os.environ.get("DEFAULT_MODEL", ""))][1] if models else None
        vae_opts = [(f"default ({os.environ.get('DEFAULT_VAE', '')})", "__default__"), ("none (use the checkpoint's)", "none")] + \
                   [(v, os.path.join(vae_dir, v)) for v in vaes]
        env = os.environ.get
        with VerticalScroll(classes="dialog form"):
            yield Label("[b]New training run[/b]", classes="dialog-title")
            with Grid(classes="fields"):
                yield Label("Dataset")
                yield Select(set_opts, value=set_opts[0][1] if set_opts else Select.NULL, allow_blank=not set_opts,
                             id="data", classes="wide")
                yield Label("Base model")
                yield Select(model_opts, value=default_model or Select.NULL, allow_blank=not model_opts, id="model",
                             classes="wide")
                yield Label("VAE")
                yield Select(vae_opts, value="__default__", allow_blank=False, id="vae", classes="wide")
                yield Label("Run name")
                yield Input(placeholder="automatic: <dataset>_<model>_d<dim>_<date-time>", id="name", classes="wide")
                yield Label("Steps")
                yield Input(env("STEPS", "1200"), type="integer", id="steps")
                yield Label("Save every")
                yield Input(env("SAVE_EVERY", "100"), type="integer", id="save")
                yield Label("Learning rate")
                yield Input(env("LR", "1e-4"), id="lr")
                yield Label("Clip skip")
                yield Select([("1  (SD 1.5 default)", 1), ("2  (most anime models)", 2)], value=1,
                             allow_blank=False, id="clip")
                yield Label("LoRA dim")
                yield Input(env("DIM", "8"), type="integer", id="dim")
                yield Label("LoRA alpha")
                yield Input(env("ALPHA", "8"), type="integer", id="alpha")
                yield Label("Resolution")
                yield Input(env("RESOLUTION", "512"), type="integer", id="res")
                yield Label("Buckets")
                yield Input(f"{env('MIN_BUCKET', '256')}:{env('MAX_BUCKET', '512')}", id="buckets")
                yield Label("Text encoder")
                yield Select([("frozen (UNet only)", False), ("trained", True)], value=False, allow_blank=False, id="te")
                yield Label("TE rate")
                yield Input(env("TE_LR", "5e-5"), id="te_lr", disabled=True)
                yield Label("VRAM limit")
                yield Select([("auto", "auto"), ("reserve MiB", "reserve"), ("cap MiB", "cap"), ("no cap", "none")],
                             value="auto", allow_blank=False, id="vram")
                yield Label("MiB")
                yield Input(placeholder="for reserve/cap", type="integer", id="vram_mib", disabled=True)
            yield Static("", id="summary", classes="summary")
            yield Static("", id="preview", classes="preview")
            with Horizontal(classes="buttons"):
                yield Button("Add to queue", variant="primary", id="add")
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        self.update_preview()

    def choices(self):
        def val(i):
            return self.query_one(f"#{i}").value
        vram, mib = val("vram"), val("vram_mib")
        vae = val("vae")
        return {
            "data": val("data") if val("data") != Select.NULL else "",
            "model": val("model") if val("model") != Select.NULL else None,
            "vae": None if vae == "__default__" else vae,
            "name": val("name").strip(), "steps": val("steps"), "save": val("save"), "lr": val("lr").strip(),
            "dim": val("dim"), "alpha": val("alpha"), "res": val("res"), "buckets": val("buckets").strip(),
            "clip_skip": val("clip"),
            "te": val("te"), "te_lr": val("te_lr").strip(),
            "vram": vram if vram in ("auto", "none") else ((vram, mib) if mib else "auto"),
        }

    def problems(self, c):
        bad = []
        if not c["data"]:
            bad.append("dataset")
        if not c["model"]:
            bad.append("base model")
        for k in ("steps", "save", "dim", "alpha", "res"):
            if not (c[k] and c[k].isdigit() and int(c[k]) > 0):
                bad.append(k)
        if not re.fullmatch(r"\d*\.?\d+(e-?\d+)?", c["lr"], re.I):
            bad.append("learning rate")
        if c["te"] and not re.fullmatch(r"\d*\.?\d+(e-?\d+)?", c["te_lr"], re.I):
            bad.append("TE rate")
        if not re.fullmatch(r"\d+:\d+", c["buckets"]):
            bad.append("buckets")
        return bad

    @on(Input.Changed)
    @on(Select.Changed)
    def update_preview(self, _event=None):
        vram = self.query_one("#vram", Select).value
        self.query_one("#vram_mib", Input).disabled = vram not in ("reserve", "cap")
        self.query_one("#te_lr", Input).disabled = not self.query_one("#te", Select).value
        c = self.choices()
        s = next((x for x in self.sets if x["name"] == c["data"]), None)
        summary = ""
        if s:
            missing = s["images"] - s["captions"]
            epochs = ""
            reps = sum(int(x.split("_")[0]) for x in s["concepts"] if x.split("_")[0].isdigit()) / max(len(s["concepts"]), 1)
            if c["steps"].isdigit() and s["images"]:
                per_epoch = s["images"] * reps / 4
                epochs = f"   about {int(c['steps']) / per_epoch:.1f} epochs at 4 images per step" if per_epoch else ""
            summary = (f"{s['images']} images in {', '.join(s['concepts']) or '(no <repeats>_<trigger>/ folder)'}"
                       + (f"   [b red]{missing} without captions[/]" if missing else "") + epochs)
            if c["steps"].isdigit():
                summary += f"\nestimated time {fmt_duration(int(c['steps']) * DEFAULT_S_PER_STEP)} at {DEFAULT_S_PER_STEP} s/step"
        self.query_one("#summary", Static).update(summary)
        bad = self.problems(c)
        import shlex
        self.query_one("#preview", Static).update(
            ("[red]check: " + ", ".join(bad) + "[/red]\n" if bad else "") +
            "[dim]gfx803-train " + " ".join(shlex.quote(a) for a in menu.args_for(c)) + "[/dim]" if c["data"] else "")
        self.query_one("#add", Button).disabled = bool(bad)

    @on(Button.Pressed)
    def done(self, event):
        if event.button.id == "add":
            c = self.choices()
            if not self.problems(c):
                job = gq.add_job(menu.args_for(c))
                self.app.notify(f"Job {job} added to the queue.", title="Training queue")
                self.dismiss(job)
                return
        self.dismiss(None)

    def key_escape(self):
        self.dismiss(None)


# ------------------------------------------------------------------ cards
class Metric(Vertical):
    """A small card: title, current value, and a sparkline of its history."""

    def __init__(self, title, unit="", id=None):
        super().__init__(id=id, classes="metric")
        self.title_text, self.unit = title, unit

    def compose(self):
        yield Label(self.title_text, classes="metric-title")
        yield Label("--", classes="metric-value")
        yield Sparkline([], summary_function=max, classes="metric-spark")

    def set(self, value_text, history, sub=""):
        self.query_one(".metric-value", Label).update(f"[b]{value_text}[/b]" + (f" [dim]{sub}[/dim]" if sub else ""))
        self.query_one(Sparkline).data = list(history)[-120:] or [0]


# ------------------------------------------------------------------ app
class Gfx803App(App):
    TITLE = "gfx803"
    SUB_TITLE = "LoRA training and ComfyUI on gfx803"
    CSS = """
    Screen { background: $surface; }
    TabPane { padding: 0 1; }
    #banner { display: none; background: $error; color: $text; padding: 0 1; text-style: bold; }
    #banner.show { display: block; }

    #metrics { height: 7; grid-size: 6; grid-gutter: 1; margin-top: 1; }
    .metric { border: round $primary 40%; padding: 0 1; height: 7; }
    .metric-title { color: $text-muted; }
    .metric-value { height: 1; }
    .metric-spark { height: 2; margin-top: 1; }
    #gpu-plot, #sys-plot { height: 1fr; min-height: 12; border: round $primary 40%; }
    #plots { height: 1fr; min-height: 14; }
    #cards { height: 11; }
    .card { border: round $accent 50%; padding: 0 1; width: 1fr; height: 11; }
    .card-title { text-style: bold; color: $accent; }
    .card ProgressBar { margin: 1 0 0 0; }
    .card Horizontal { height: 3; }
    .card Button { min-width: 10; margin-right: 1; }

    .section { text-style: bold; color: $accent; margin-top: 1; }
    DataTable { height: 1fr; min-height: 6; }
    #queue-table { height: 1fr; }
    #runs-table { height: 1fr; }
    .toolbar { height: 3; }
    .toolbar Button { min-width: 8; margin-right: 1; }
    .toolbar Input { width: 12; }
    .toolbar Label { padding: 1 1 0 2; }

    #comfy-status { height: auto; border: round $accent 50%; padding: 0 1; }
    #comfy-log { height: 1fr; border: round $primary 40%; }
    .switch-row { height: 3; }
    .switch-row Label { padding: 1 1 0 1; }

    #cores { height: auto; }
    #proc-table { height: 1fr; }

    .dialog { width: 72; height: auto; max-height: 90%; background: $panel; border: thick $accent;
              padding: 1 2; }
    .dialog.wide { width: 95%; height: 90%; }
    .dialog.form { width: 120; }
    .dialog-text { margin-bottom: 1; width: 100%; }
    .dialog-title { margin-bottom: 1; color: $accent; }
    .fields { grid-size: 4; grid-columns: 15 1fr 13 1fr; grid-gutter: 0 1; height: auto; grid-rows: 3; }
    .fields Label { padding: 1 0 0 0; }
    .fields .wide { column-span: 3; }
    .copy-fields { grid-size: 2; grid-columns: 14 1fr; }
    .dialog.form-small { width: 90; }
    #ds-left { width: 46; }
    #ds-right { padding-left: 1; }
    #ds-table { height: 1fr; }
    #ds-detail { height: auto; border: round $accent 50%; padding: 0 1; }
    #tag-table { height: 1fr; }
    #gen-form { width: 74; padding-right: 1; }
    .gen-fields { grid-size: 4; grid-columns: 11 1fr 10 1fr; grid-rows: 3; height: auto; }
    .gen-fields .wide { column-span: 3; }
    .gen-fields .span2 { column-span: 2; }
    .gen-fields .strength { width: 100%; }
    #g-pos { height: 6; }
    #g-neg { height: 4; }
    #gen-right { padding-left: 1; }
    #g-status { height: auto; border: round $accent 50%; padding: 0 1; }
    #g-queue-table { height: 8; }
    #g-gal { height: 1fr; }
    #g-gallery { width: 46; height: 1fr; }
    #g-gal-right { padding-left: 1; }
    #g-prev-box { height: 1fr; align: center middle; }
    #g-prev { width: auto; height: auto; }
    #g-prev-info { height: auto; max-height: 4; }
    #cmp-grid { height: 1fr; grid-gutter: 1; }
    .cmp-cell { height: 1fr; }
    .cmp-img { height: 1fr; align: center middle; }
    .cmp-img > * { width: auto; height: auto; }
    .cmp-label { width: 100%; content-align: center middle; height: 1; }
    #cmp-desc { height: auto; padding: 0 1; background: $panel; width: 100%; }
    #cmp-list { height: 12; }
    #ib-list { width: 36; }
    #ib-images { height: 1fr; }
    #ib-preview { width: 1fr; padding: 0 1; }
    #ib-img-box { height: 1fr; align: center middle; }
    #ib-img, #full-img { width: auto; height: auto; }
    #ib-info { height: 1; }
    #ib-caption { width: 44; }
    #ib-tags { height: 1fr; }
    .hint { height: auto; width: 100%; }
    #full-box { height: 1fr; align: center middle; }
    #full-title { height: 1; padding: 0 1; background: $panel; width: 100%; }
    #full-caption { height: auto; max-height: 4; padding: 0 1; background: $panel; width: 100%; }
    .summary { margin-top: 1; color: $text-muted; }
    .preview { margin-top: 1; }
    .buttons { height: 3; margin-top: 1; align-horizontal: right; }
    .buttons Button { margin-left: 1; }
    ModalScreen { align: center middle; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit (service keeps running)"),
        Binding("a", "add_run", "Add run"),
        Binding("c", "toggle_comfy", "ComfyUI on/off"),
        Binding("p", "toggle_pause", "Pause/resume queue"),
        Binding("ctrl+x", "shutdown", "Shut down service"),
        Binding("1", "tab('overview')", show=False),
        Binding("2", "tab('training')", show=False),
        Binding("3", "tab('datasets')", show=False),
        Binding("4", "tab('generate')", show=False),
        Binding("5", "tab('comfyui')", show=False),
        Binding("6", "tab('system')", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.dev = st.gpu_device()
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        self.prev_sys = None
        self.q = gq.empty_queue()
        self.comfy_log_size = -1
        self.ds_info = {}
        self.ds_transparent = {}
        self.tagging = None
        self.gen_meta = {}
        self.gen_waiting = []
        self.gen_progress = {}
        self.gen_gallery = []

    # ---- layout
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="banner")
        with TabbedContent(initial="overview"):
            with TabPane("Overview [1]", id="overview"):
                with Grid(id="metrics"):
                    yield Metric("GPU load", id="m-busy")
                    yield Metric("GPU power", id="m-power")
                    yield Metric("GPU temp", id="m-temp")
                    yield Metric("VRAM", id="m-vram")
                    yield Metric("CPU", id="m-cpu")
                    yield Metric("RAM", id="m-ram")
                with Horizontal(id="plots"):
                    yield PlotextPlot(id="gpu-plot")
                    yield PlotextPlot(id="sys-plot")
                with Horizontal(id="cards"):
                    with Vertical(classes="card", id="train-card"):
                        yield Label("Training", classes="card-title")
                        yield Label("", id="train-line1")
                        yield ProgressBar(total=100, show_eta=False, id="train-progress")
                        yield Label("", id="train-line2")
                        with Horizontal():
                            yield Button("Add run", id="ov-add", variant="primary")
                            yield Button("Pause", id="ov-pause")
                            yield Button("Stop job", id="ov-stop", variant="error")
                    with Vertical(classes="card", id="comfy-card"):
                        yield Label("ComfyUI", classes="card-title")
                        yield Label("", id="comfy-line1")
                        yield Label("", id="comfy-line2")
                        with Horizontal():
                            yield Button("Start", id="ov-comfy", variant="success")
            with TabPane("Training [2]", id="training"):
                yield Label("Queue", classes="section")
                with Horizontal(classes="toolbar"):
                    yield Button("Add run", id="q-add", variant="primary")
                    yield Button("Up", id="q-up")
                    yield Button("Down", id="q-down")
                    yield Button("Remove", id="q-remove")
                    yield Button("Retry", id="q-retry")
                    yield Button("Skip job", id="q-skip")
                    yield Button("Stop job", id="q-stop", variant="error")
                    yield Button("Pause", id="q-pause")
                    yield Label("Cool-down s")
                    yield Input("0", type="integer", id="q-cooldown")
                yield Label("[dim]The queue is empty. Press [b]a[/b] or Add run to queue a training run; queued runs "
                            "train one after another.[/dim]", id="queue-empty")
                yield DataTable(id="queue-table", cursor_type="row", zebra_stripes=True)
                yield Label("Runs", classes="section")
                with Horizontal(classes="toolbar"):
                    yield Button("View log", id="r-log")
                    yield Button("Copy LoRA to ComfyUI", id="r-copy", variant="success")
                    yield Button("Resume", id="r-resume")
                yield DataTable(id="runs-table", cursor_type="row", zebra_stripes=True)
            with TabPane("Datasets [3]", id="datasets"):
                with Horizontal():
                    with Vertical(id="ds-left"):
                        yield Label("Datasets  [dim]workspace/training_data[/dim]", classes="section")
                        yield DataTable(id="ds-table", cursor_type="row", zebra_stripes=True)
                    with Vertical(id="ds-right"):
                        yield Label("", id="ds-detail")
                        with Horizontal(classes="toolbar"):
                            yield Button("Browse images", id="d-browse", variant="success")
                            yield Button("Tag with WD14", id="d-tag", variant="primary")
                            yield Button("Set trigger", id="d-trigger")
                            yield Button("Remove tags", id="d-remove")
                            yield Button("Flatten transparent", id="d-flatten")
                            yield Button("Undo last change", id="d-undo", variant="warning")
                            yield Button("Tagger log", id="d-log")
                        yield Label("Tags  [dim](Enter on a tag: browse the images that have it; D or Delete: "
                                    "remove it from every caption)[/dim]",
                                    classes="section")
                        yield TagTable(id="tag-table", cursor_type="row", zebra_stripes=True)
            with TabPane("Generate [4]", id="generate"):
                with Horizontal():
                    with VerticalScroll(id="gen-form"):
                        with Horizontal(classes="toolbar"):
                            yield Select([], prompt="presets", id="g-preset")
                            yield Button("Load", id="g-load")
                            yield Button("Save", id="g-save")
                            yield Button("Save as", id="g-saveas")
                            yield Button("Delete", id="g-delete", variant="error")
                        with Grid(classes="fields gen-fields"):
                            yield Label("Checkpoint")
                            yield Select([], id="g-ckpt", classes="wide")
                            yield Label("VAE")
                            yield Select([], id="g-vae", classes="wide")
                            for i in range(3):
                                yield Label(f"LoRA {i + 1}")
                                yield Select([], id=f"g-lora{i}", classes="span2")
                                yield Input("1.0", id=f"g-lstr{i}", classes="strength")
                            yield Label("Size")
                            yield Select([(f"{w}x{h}", f"{w}x{h}") for w, h in gen.SIZES], value="512x512",
                                         allow_blank=False, id="g-size")
                            yield Label("Clip skip")
                            yield Select([("1", 1), ("2 (anime)", 2)], value=1, allow_blank=False, id="g-clip")
                            yield Label("Steps")
                            yield Input("25", type="integer", id="g-steps")
                            yield Label("CFG")
                            yield Input("7.0", id="g-cfg")
                            yield Label("Sampler")
                            yield Select([(x, x) for x in gen.SAMPLERS], value="dpmpp_2m", allow_blank=False, id="g-sampler")
                            yield Label("Scheduler")
                            yield Select([(x, x) for x in gen.SCHEDULERS], value="karras", allow_blank=False,
                                         id="g-scheduler")
                            yield Label("Seed")
                            yield Input("-1", id="g-seed", placeholder="-1 = random")
                            yield Label("Images")
                            yield Input("1", type="integer", id="g-count")
                        yield Label("Prompt")
                        yield TextArea(id="g-pos", soft_wrap=True, show_line_numbers=False)
                        yield Label("Negative prompt")
                        yield TextArea(id="g-neg", soft_wrap=True, show_line_numbers=False)
                        with Horizontal(classes="buttons"):
                            yield Button("Compare LoRA checkpoints", id="g-compare")
                            yield Button("Queue", id="g-queue", variant="primary")
                    with Vertical(id="gen-right"):
                        yield Label("", id="g-status")
                        yield Label("Queue  [dim](ComfyUI's queue, including the web UI's)[/dim]", classes="section")
                        yield DataTable(id="g-queue-table", cursor_type="row", zebra_stripes=True)
                        with Horizontal(classes="toolbar"):
                            yield Button("Cancel selected", id="g-cancel", variant="error")
                        yield Label("Gallery  [dim]ComfyUI/output/gfx803[/dim]", classes="section")
                        with Horizontal(id="g-gal"):
                            yield DataTable(id="g-gallery", cursor_type="row", zebra_stripes=True)
                            with Vertical(id="g-gal-right"):
                                with Container(id="g-prev-box"):
                                    yield PreviewImage(id="g-prev")
                                yield Label("", id="g-prev-info")
                        with Horizontal(classes="toolbar"):
                            yield Button("Full screen", id="g-full")
                            yield Button("Use these settings", id="g-use")
                            yield Button("Show comparison", id="g-showcmp")
            with TabPane("ComfyUI [5]", id="comfyui"):
                with Vertical(id="comfy-status"):
                    yield Label("", id="comfy-detail")
                    with Horizontal(classes="switch-row"):
                        yield Button("Start ComfyUI", id="c-toggle", variant="success")
                        yield Label("Force fp32")
                        yield Switch(id="c-fp32")
                        yield Label("[dim]fp16 (ComfyUI's default on ROCm) is not validated on gfx803; "
                                    "fp32 is. Applies at the next start.[/dim]")
                yield Label("[dim]ComfyUI's output appears here once it has been started.[/dim]", id="comfy-log-empty")
                yield Log(id="comfy-log", highlight=False, max_lines=3000)
            with TabPane("System [6]", id="system"):
                yield Label("", id="sys-summary")
                yield Static("", id="cores")
                yield Label("Top processes (host)", classes="section")
                yield DataTable(id="proc-table", zebra_stripes=True)
        yield Footer()

    def on_mount(self):
        load_defaults()
        self.query_one("#queue-table", DataTable).add_columns("ID", "State", "Job", "Progress", "Note")
        self.query_one("#runs-table", DataTable).add_columns("Run", "Dataset", "Model", "State", "Steps", "Loss", "Updated")
        self.query_one("#proc-table", DataTable).add_columns("PID", "User", "CPU %", "RAM", "Command")
        self.query_one("#ds-table", DataTable).add_columns("Dataset", "Images", "Captioned")
        self.query_one("#tag-table", DataTable).add_columns("Tag", "Images", "%")
        self.refresh_datasets()
        self.setup_generate()
        self.comfy_events()
        for pid in ("#gpu-plot", "#sys-plot"):
            self.query_one(pid, PlotextPlot).plt.theme("textual-design-dark")
        psutil.cpu_percent(percpu=True)
        self.tick()
        self.set_interval(1.0, self.tick)
        self.set_interval(3.0, self.slow_tick)
        self.slow_tick()

    # ---- periodic updates
    def tick(self):
        self.q = gq.load()
        g = st.gpu_sample(self.dev)
        s = st.system_sample(psutil)
        h = self.hist
        for k, v in (("busy", g.get("busy")), ("power", g.get("power_w")), ("temp", g.get("temp_c")),
                     ("vram", g.get("vram_used_gib")), ("sclk", g.get("sclk_mhz")), ("cpu", s["cpu"]),
                     ("ram", s["ram_used_gib"])):
            h[k].append(v if v is not None else 0)
        h["t"].append(time.time())
        self.sys = s
        self.update_metrics(g, s)
        self.update_plots(g, s)
        self.update_cards()
        self.update_banner()
        self.update_queue_table()
        self.update_comfy_tab()

    def slow_tick(self):
        self.update_runs_table()
        self.update_system_tab()
        self.poll_tagging()
        self.update_generate()
        try:
            names = sorted(d for d in os.listdir(ds.DATA_ROOT) if os.path.isdir(os.path.join(ds.DATA_ROOT, d)))
        except OSError:
            names = []
        if names != getattr(self, "ds_names", None):
            self.ds_names = names
            self.refresh_datasets()

    def update_banner(self):
        svc = self.q.get("service") or {}
        stale = not svc.get("heartbeat") or time.time() - svc["heartbeat"] > SERVICE_STALE_S
        banner = self.query_one("#banner", Static)
        if stale:
            banner.update("gfx803-service is not running: queued jobs will not start and ComfyUI cannot be "
                          "started. Start it with the gfx803 launcher on the host.")
        banner.set_class(stale, "show")

    def update_metrics(self, g, s):
        h = self.hist
        def num(v, fmt):
            return fmt.format(v) if v is not None else "--"
        self.query_one("#m-busy", Metric).set(num(g.get("busy"), "{:.0f} %"), h["busy"],
                                              num(g.get("sclk_mhz"), "{} MHz"))
        self.query_one("#m-power", Metric).set(num(g.get("power_w"), "{:.0f} W"), h["power"],
                                               num(g.get("power_cap_w"), "cap {:.0f} W"))
        self.query_one("#m-temp", Metric).set(num(g.get("temp_c"), "{:.0f} °C"), h["temp"],
                                              num(g.get("fan_rpm"), "fan {} rpm"))
        self.query_one("#m-vram", Metric).set(num(g.get("vram_used_gib"), "{:.1f} GiB"), h["vram"],
                                              num(g.get("vram_total_gib"), "of {:.0f}"))
        self.query_one("#m-cpu", Metric).set(f"{s['cpu']:.0f} %", h["cpu"],
                                             num(s.get("cpu_temp_c"), "{:.0f} °C"))
        self.query_one("#m-ram", Metric).set(f"{s['ram_used_gib']:.1f} GiB", h["ram"],
                                             f"of {s['ram_total_gib']:.0f}")

    @staticmethod
    def time_axis(p, now):
        # Ticks sit on whole two-minute marks of the local clock so their labels
        # stay put while the data scrolls past them.
        first = math.ceil((now - HISTORY) / 120) * 120
        marks = range(first, int(now) + 1, 120)
        p.xlim(-HISTORY, 0)
        p.xticks([m - now for m in marks], [time.strftime("%H:%M", time.localtime(m)) for m in marks])

    def update_plots(self, g, s):
        now = time.time()
        xs = [t - now for t in self.hist["t"]]
        gp = self.query_one("#gpu-plot", PlotextPlot)
        p = gp.plt
        p.clear_data()
        p.title("GPU  (last 10 min)")
        self.time_axis(p, now)
        cap = g.get("power_cap_w") or 185
        p.plot(xs, list(self.hist["busy"]), label="load %", color="cyan", marker="braille")
        p.plot(xs, [w / cap * 100 for w in self.hist["power"]], label="power % of cap", color="orange", marker="braille")
        total = g.get("vram_total_gib") or 8
        p.plot(xs, [v / total * 100 for v in self.hist["vram"]], label="VRAM %", color="magenta", marker="braille")
        p.ylim(0, 100)
        gp.refresh()
        sp = self.query_one("#sys-plot", PlotextPlot)
        p = sp.plt
        p.clear_data()
        p.title("System")
        self.time_axis(p, now)
        p.plot(xs, list(self.hist["cpu"]), label="CPU %", color="green", marker="braille")
        p.plot(xs, [r / s["ram_total_gib"] * 100 for r in self.hist["ram"]], label="RAM %", color="blue", marker="braille")
        p.plot(xs, list(self.hist["temp"]), label="GPU °C", color="red", marker="braille")
        p.ylim(0, 100)
        sp.refresh()

    def queue_eta(self, s_per_step):
        total = 0
        for j in self.q["jobs"]:
            if j["state"] == gq.PENDING:
                total += self.job_steps(j)
        cur = gq.current_job(self.q)
        if cur:
            prog = st.run_progress(cur["run"])
            total += prog["total"] - prog["step"] if prog else self.job_steps(cur)
        return total * s_per_step

    def job_steps(self, job):
        args = job["args"]
        if "--resume-run" in args:
            run = args[args.index("--resume-run") + 1]
            steps = int(gq.read_kv(os.path.join(gq.RUNS_DIR, run, "run.env")).get("STEPS", "1200") or 1200)
            saved = [s for s, _ in st.checkpoints(run) if s]
            return max(steps - max(saved, default=0), 0)
        if "--steps" in args:
            return int(args[args.index("--steps") + 1])
        return int(os.environ.get("STEPS", "1200"))

    def update_cards(self):
        q = self.q
        cur = gq.current_job(q)
        pending = sum(1 for j in q["jobs"] if j["state"] == gq.PENDING)
        bar = self.query_one("#train-progress", ProgressBar)
        sps = DEFAULT_S_PER_STEP
        line2 = ""
        if cur:
            prog = st.run_progress(cur["run"])
            line1 = f"[b cyan]job {cur['id']}[/]  {cur['label']}"
            if prog:
                sps = prog["s_per_step"] or sps
                bar.update(total=prog["total"], progress=prog["step"])
                line2 = (f"step {prog['step']}/{prog['total']}   {sps:.2f} s/step   loss {prog['loss'] or '?'}   "
                         f"ETA {fmt_eta((prog['total'] - prog['step']) * sps)}")
            else:
                bar.update(total=None)
                line2 = "[dim]preparing: preflight, latents, MIOpen setup...[/dim]"
        else:
            bar.update(total=100, progress=0)
            why = ""
            if q["paused"]:
                why = f"[yellow]queue paused[/] ({q['pause_reason']})"
            elif pending and q.get("comfyui", {}).get("want"):
                why = "[yellow]waiting: ComfyUI has the GPU[/]"
            line1 = "[dim]nothing training[/dim]" + ("   " + why if why else "")
        summary = f"{pending} queued" + (f", all done in about {fmt_duration(self.queue_eta(sps))}" if pending or cur else "")
        self.query_one("#train-line1", Label).update(line1)
        self.query_one("#train-line2", Label).update((line2 + "\n" if line2 else "") + f"[dim]{summary}[/dim]")
        self.query_one("#ov-pause", Button).label = "Resume" if q["paused"] else "Pause"
        self.query_one("#q-pause", Button).label = "Resume" if q["paused"] else "Pause"
        self.query_one("#ov-stop", Button).disabled = cur is None

        c = q.get("comfyui") or {}
        state = c.get("state", "stopped")
        colors = {"running": "green", "starting": "yellow", "waiting": "yellow", "stopping": "yellow", "stopped": "dim"}
        self.query_one("#comfy-line1", Label).update(f"[{colors.get(state, '')}]● {state}[/]" + ("   fp32" if c.get("fp32") else "   fp16 (ComfyUI default)")
            + (f"   [yellow]{c['note']}[/]" if c.get("note") else ""))
        urls = "  ".join(comfy_urls())
        self.query_one("#comfy-line2", Label).update(f"[b]{urls}[/b]" if state == "running" else f"[dim]{urls}[/dim]")
        wanted = c.get("want")
        for bid in ("#ov-comfy", "#c-toggle"):
            b = self.query_one(bid, Button)
            b.label = ("Stop" if bid == "#ov-comfy" else "Stop ComfyUI") if wanted else ("Start" if bid == "#ov-comfy" else "Start ComfyUI")
            b.variant = "error" if wanted else "success"

    def selected_key(self, table_id):
        t = self.query_one(table_id, DataTable)
        if not t.row_count:
            return None
        try:
            return t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
        except Exception:
            return None

    def update_queue_table(self):
        t = self.query_one("#queue-table", DataTable)
        keep = self.selected_key("#queue-table")
        t.clear()
        for j in self.q["jobs"]:
            prog = ""
            if j["state"] == gq.RUNNING:
                p = st.run_progress(j["run"])
                prog = f"{p['step']}/{p['total']}  loss {p['loss'] or '?'}" if p else "starting"
            elif j["state"] == gq.PENDING:
                prog = f"{self.job_steps(j)} steps"
            style = STATE_STYLE.get(j["state"], "")
            t.add_row(str(j["id"]), f"[{style}]{j['state']}[/]" if style else j["state"], j["label"], prog,
                      j.get("note", ""), key=str(j["id"]))
        if keep is not None:
            for i, j in enumerate(self.q["jobs"]):
                if str(j["id"]) == keep:
                    t.move_cursor(row=i)
        self.query_one("#queue-empty", Label).display = not self.q["jobs"]
        cd = self.query_one("#q-cooldown", Input)
        if not cd.has_focus and cd.value != str(self.q["cooldown_s"]):
            cd.value = str(self.q["cooldown_s"])

    def update_runs_table(self):
        t = self.query_one("#runs-table", DataTable)
        keep = self.selected_key("#runs-table")
        t.clear()
        self.run_list = st.runs()
        for r in self.run_list:
            style = {"completed": "green", "failed": "red", "nan": "red", "interrupted": "yellow",
                     "running": "cyan"}.get(r["state"], "")
            steps = f"{r['step'] or 0}/{r['steps']}"
            upd = time.strftime("%m-%d %H:%M", time.localtime(r["updated"])) if r["updated"] else "-"
            t.add_row(r["name"], r["data"], r["model"], f"[{style}]{r['state']}[/]" if style else r["state"],
                      steps, r["loss"] or "-", upd, key=r["name"])
        if keep:
            for i, r in enumerate(self.run_list):
                if r["name"] == keep:
                    t.move_cursor(row=i)

    def update_comfy_tab(self):
        c = self.q.get("comfyui") or {}
        up = f"   up {fmt_duration(time.time() - c['started'])}" if c.get("state") == "running" and c.get("started") else ""
        urls = ", ".join(comfy_urls())
        detail = (f"[b]ComfyUI[/b]  state [b]{c.get('state', 'stopped')}[/b]{up}"
                  + (f"   [yellow]{c['note']}[/]" if c.get("note") else "")
                  + f"\nOpen from another machine on the LAN: [b]{urls}[/b]"
                  + "\nTraining and ComfyUI never use the GPU at the same time: starting ComfyUI holds new "
                    "training jobs, and waits for a job that is running.")
        self.query_one("#comfy-detail", Label).update(detail)
        sw = self.query_one("#c-fp32", Switch)
        if sw.value != bool(c.get("fp32")) and not sw.has_focus:
            sw.value = bool(c.get("fp32"))
        try:
            size = os.path.getsize(os.path.join(gq.QUEUE_DIR, "comfyui.log"))
        except OSError:
            size = 0
        self.query_one("#comfy-log-empty", Label).display = size == 0
        if size != self.comfy_log_size:
            self.comfy_log_size = size
            log = self.query_one("#comfy-log", Log)
            log.clear()
            log.write_lines([l for l in st.tail(os.path.join(gq.QUEUE_DIR, "comfyui.log"), 100000).split("\n") if l.strip()][-1500:])

    def update_system_tab(self):
        s = getattr(self, "sys", None)
        if not s:
            return
        now = s["time"]
        rates = ""
        if self.prev_sys:
            dt = max(now - self.prev_sys["time"], 0.1)
            mb = lambda a, b: (s[a] - self.prev_sys[a]) / dt / 2**20
            rates = (f"   net ↓ {mb('net_rx', None):.2f} ↑ {mb('net_tx', None):.2f} MiB/s"
                     f"   disk r {mb('disk_read', None):.1f} w {mb('disk_write', None):.1f} MiB/s")
        self.prev_sys = s
        la = s["load"]
        self.query_one("#sys-summary", Label).update(f"[b]CPU[/b] {s['cpu']:.0f} %  load {la[0]:.2f} {la[1]:.2f} {la[2]:.2f}"
            + (f"  {s['cpu_temp_c']:.0f} °C" if s.get("cpu_temp_c") else "")
            + f"   [b]RAM[/b] {s['ram_used_gib']:.1f}/{s['ram_total_gib']:.0f} GiB  swap {s['swap_used_gib']:.1f} GiB"
            + (f"   [b]workspace disk[/b] {s['ws_used_gib']:.0f}/{s['ws_total_gib']:.0f} GiB" if s.get("ws_total_gib") else "")
            + rates)
        bars = []
        for i, pct in enumerate(s["cpu_per_core"]):
            filled = int(pct / 100 * 20)
            color = "green" if pct < 60 else "yellow" if pct < 90 else "red"
            bars.append(f"cpu{i:<2} [{color}]{'█' * filled}[/][dim]{'░' * (20 - filled)}[/] {pct:3.0f}%")
        cols = 4
        rows = [bars[i:i + cols] for i in range(0, len(bars), cols)]
        self.query_one("#cores", Static).update("\n".join("   ".join(r) for r in rows))
        t = self.query_one("#proc-table", DataTable)
        procs = []
        for p in psutil.process_iter(["pid", "username", "cpu_percent", "memory_info", "name", "cmdline"]):
            info = p.info
            procs.append(info)
        procs.sort(key=lambda i: (i.get("cpu_percent") or 0, (i.get("memory_info").rss if i.get("memory_info") else 0)), reverse=True)
        t.clear()
        for info in procs[:25]:
            mem = info.get("memory_info")
            cmd = " ".join(info.get("cmdline") or []) or info.get("name") or ""
            t.add_row(str(info["pid"]), info.get("username") or "?", f"{info.get('cpu_percent') or 0:.1f}",
                      f"{mem.rss / 2**20:.0f} MiB" if mem else "-", cmd[:140])

    # ---- datasets
    def ds_path(self, name):
        return os.path.join(ds.DATA_ROOT, name) if name else None

    def refresh_datasets(self):
        t = self.query_one("#ds-table", DataTable)
        keep = self.selected_key("#ds-table")
        t.clear()
        sets = menu.datasets(ds.DATA_ROOT)
        for x in sets:
            cap = x["captions"]
            cap_txt = f"{cap}" if cap == x["images"] else f"[yellow]{cap}[/]"
            t.add_row(x["name"], str(x["images"]), cap_txt, key=x["name"])
        names = [x["name"] for x in sets]
        if keep in names:
            t.move_cursor(row=names.index(keep))
        self.update_dataset_detail()

    @on(DataTable.RowHighlighted, "#ds-table")
    def dataset_selected(self, _event):
        self.update_dataset_detail()

    def update_dataset_detail(self):
        name = self.selected_key("#ds-table")
        detail = self.query_one("#ds-detail", Label)
        tags = self.query_one("#tag-table", DataTable)
        tags.clear()
        if not name:
            detail.update("[dim]No datasets. Put sd-scripts datasets (folders of <repeats>_<trigger>/ subfolders "
                          "with images) under workspace/training_data/.[/dim]")
            return
        path = self.ds_path(name)
        try:
            info = ds.summary(path)
        except OSError as e:
            detail.update(f"[red]{e}[/red]")
            return
        self.ds_info[name] = info
        trans = self.ds_transparent.get(name)
        if trans is None:
            self.scan_transparency(name)
            trans_txt = "[dim]checking...[/dim]"
        else:
            trans_txt = f"[yellow]{len(trans)}[/]" if trans else "none"
        bks = ds.backups(path)
        tagging = ""
        if self.tagging and self.tagging["set"] == name:
            tagging = f"\n[b cyan]tagging in progress[/] ({self.tagging['count']} images, " \
                      f"{fmt_duration(time.time() - self.tagging['started'])})"
        detail.update(
            f"[b]{name}[/b]  [dim]{path}[/dim]\n"
            f"folders {', '.join(info['concepts']) or '[red]none (needs <repeats>_<trigger>/)[/red]'}   "
            f"images {info['images']}   captioned {info['captioned']}"
            + (f"   [yellow]{info['uncaptioned']} without caption[/]" if info["uncaptioned"] else "")
            + f"\ntrigger (first tag of every caption) {info['trigger'] or '[yellow]none[/]'}   "
            f"transparent images {trans_txt}   undo steps {len(bks)}"
            + (f" [dim](last: {bks[0][1]['action']})[/dim]" if bks else "") + tagging)
        n = max(info["captioned"], 1)
        for tag, count in info["freq"].most_common():
            style = "dim" if tag == info["trigger"] else ("yellow" if count == info["captioned"] else "")
            label = f"[{style}]{tag}[/]" if style else tag
            tags.add_row(label, str(count), f"{100 * count / n:.0f}", key=tag)

    @on(DataTable.RowSelected, "#ds-table")
    def dataset_opened(self, _event):
        self.dataset_action("browse")

    @on(DataTable.RowSelected, "#tag-table")
    def tag_opened(self, event):
        name = self.selected_key("#ds-table")
        if name and event.row_key.value:
            self.push_screen(ImageBrowser(name, event.row_key.value))

    @work(thread=True, exclusive=True, group="transparency")
    def scan_transparency(self, name):
        found = ds.transparent_images(self.ds_path(name))
        self.call_from_thread(self._transparency_done, name, found)

    def _transparency_done(self, name, found):
        self.ds_transparent[name] = found
        if self.selected_key("#ds-table") == name:
            self.update_dataset_detail()

    def dataset_changed(self, name, message):
        self.ds_transparent.pop(name, None)
        self.refresh_datasets()
        self.notify(message, title=name)

    def poll_tagging(self):
        t = self.tagging
        if not t or t["proc"].poll() is None:
            if t and self.selected_key("#ds-table") == t["set"]:
                self.update_dataset_detail()
            return
        self.tagging = None
        rc = t["proc"].returncode
        moved, total = t["finish"]()
        if rc == 0:
            self.dataset_changed(t["set"], f"Tagged {moved} of {total} images.")
        else:
            self.dataset_changed(t["set"], f"The tagger failed (exit {rc}); see Tagger log. {moved} captions written.")

    def action_remove_tag(self):
        self.dataset_action("remove")

    @work
    async def dataset_action(self, what):
        name = self.selected_key("#ds-table")
        if not name:
            self.notify("Select a dataset first.", severity="warning")
            return
        path = self.ds_path(name)
        info = self.ds_info.get(name) or ds.summary(path)
        if what == "browse":
            self.push_screen(ImageBrowser(name))
            return
        if what == "log":
            self.push_screen(LogView("Tagger log", os.path.join(gq.QUEUE_DIR, "tagger.log")))
            return
        if self.tagging and what != "log":
            self.notify("Wait for the tagging to finish first.", severity="warning")
            return
        if what == "tag":
            if not info["images"]:
                self.notify("This dataset has no images in <repeats>_<trigger>/ folders.", severity="warning")
                return
            opts = await self.push_screen_wait(TagDialog(name, info, info["trigger"] or ds.suggested_trigger(path)))
            if not opts:
                return
            os.makedirs(gq.QUEUE_DIR, exist_ok=True)
            proc, finish = ds.start_tagging(path, opts["repo"], opts["general"], opts["character"], opts["mode"],
                                            opts["trigger"], os.path.join(gq.QUEUE_DIR, "tagger.log"))
            count = info["uncaptioned"] if opts["mode"] == "untagged" else info["images"]
            self.tagging = {"set": name, "proc": proc, "finish": finish, "started": time.time(), "count": count}
            self.notify(f"Tagging {count} images on the CPU.", title=name)
            self.update_dataset_detail()
        elif what == "trigger":
            value = await self.push_screen_wait(TextDialog(
                "Set trigger word", "Put this tag first in every caption, creating captions that are missing, "
                "and drop other copies of it. The trigger is what you type in a prompt to call up the LoRA.",
                info["trigger"] or ds.suggested_trigger(path), "e.g. mychar", ok="Apply"))
            if value and value.strip():
                n = ds.set_trigger(path, value)
                self.dataset_changed(name, f"Trigger '{value.strip()}' set in {n} captions.")
        elif what == "remove":
            highlighted = self.selected_key("#tag-table")
            pre = highlighted if highlighted and highlighted != info["trigger"] else ""
            value = await self.push_screen_wait(TextDialog(
                "Remove tags", "Comma-separated tags to delete from every caption. Prefilled with the highlighted "
                "tag. The trigger is kept.", pre, "e.g. 1girl, solo", ok="Remove"))
            if value:
                c, r = ds.remove_tags(path, value.split(","), keep=[info["trigger"]] if info["trigger"] else [])
                self.dataset_changed(name, f"Removed {r} tags from {c} captions.")
        elif what == "flatten":
            trans = self.ds_transparent.get(name)
            if trans is None:
                self.notify("Still checking for transparency; try again in a moment.", severity="warning")
                return
            if not trans:
                self.notify("No transparent images in this dataset.")
                return
            names = ", ".join(os.path.basename(p) for p in trans[:5]) + (" ..." if len(trans) > 5 else "")
            if await self.push_screen_wait(Confirm(
                    f"Flatten {len(trans)} transparent image(s) onto white?\n{names}\n\nTraining converts images to "
                    "RGB, which leaves transparent areas whatever color their pixels hold, often black. "
                    "The originals are backed up; Undo restores them.", ok="Flatten")):
                n = ds.flatten(path, trans)
                self.dataset_changed(name, f"Flattened {n} image(s) onto white.")
        elif what == "undo":
            bks = ds.backups(path)
            if not bks:
                self.notify("Nothing to undo for this dataset.")
                return
            m = bks[0][1]
            when = time.strftime("%m-%d %H:%M", time.localtime(m["time"]))
            if await self.push_screen_wait(Confirm(
                    f"Undo '{m['action']}' from {when}? It restores {len(m['saved'])} file(s) and removes "
                    f"{len(m['created'])} it created.", ok="Undo", danger=True)):
                ds.undo_last(path)
                self.dataset_changed(name, f"Undid '{m['action']}'.")

    @work
    async def action_quit(self):
        if self.tagging and not await self.push_screen_wait(Confirm(
                "Tagging is still running and its captions are put in place by this app when it finishes. "
                "Quit anyway? The tagging is lost.", ok="Quit", danger=True)):
            return
        if self.tagging:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.tagging["proc"].pid, signal.SIGTERM)
        self.exit()

    # ---- generate
    NONE = "__none__"

    def setup_generate(self):
        self.query_one("#g-queue-table", DataTable).add_columns("Request", "State", "Progress")
        self.query_one("#g-gallery", DataTable).add_columns("Time", "Image", "Seed")
        self.refresh_model_lists()
        self.refresh_presets()
        self.apply_settings(gen.last_settings())

    def refresh_model_lists(self):
        ckpts = gen.model_files("checkpoints")
        self.query_one("#g-ckpt", Select).set_options([(c, c) for c in ckpts])
        self.query_one("#g-vae", Select).set_options(
            [("(from checkpoint)", self.NONE)] + [(v, v) for v in gen.model_files("vae")])
        loras = [("(none)", self.NONE)] + [(l, l) for l in gen.model_files("loras")]
        for run, cks in gen.training_loras().items():
            for step, name in reversed(cks):
                loras.append((f"training: {run} {'step ' + str(step) if step else 'final'}", name))
        self.lora_options = loras
        for i in range(3):
            self.query_one(f"#g-lora{i}", Select).set_options(loras)

    def refresh_presets(self, select=None):
        names = sorted(gen.presets(), key=str.lower)
        sel = self.query_one("#g-preset", Select)
        sel.set_options([(n, n) for n in names])
        if select in names:
            sel.value = select

    def _set_select(self, sid, value, fallback=None):
        sel = self.query_one(sid, Select)
        values = [v for _l, v in sel._options if v is not Select.BLANK]
        if value in values:
            sel.value = value
        elif fallback is not None and fallback in values:
            sel.value = fallback
        elif values and sel.value is Select.BLANK:
            sel.value = values[0]
        return value in values

    def apply_settings(self, st_):
        missing = []
        if st_.get("checkpoint") and not self._set_select("#g-ckpt", st_["checkpoint"]):
            missing.append(st_["checkpoint"])
        self._set_select("#g-vae", st_.get("vae") or self.NONE, self.NONE)
        loras = [l for l in st_.get("loras", []) if l.get("name")]
        for i in range(3):
            l = loras[i] if i < len(loras) else {}
            if l and not self._set_select(f"#g-lora{i}", l["name"], self.NONE):
                missing.append(l["name"])
            elif not l:
                self._set_select(f"#g-lora{i}", self.NONE)
            self.query_one(f"#g-lstr{i}", Input).value = str(l.get("strength", 1.0)) if l else "1.0"
        self._set_select("#g-size", f"{st_.get('width', 512)}x{st_.get('height', 512)}", "512x512")
        self._set_select("#g-clip", int(st_.get("clip_skip", 1)), 1)
        self._set_select("#g-sampler", st_.get("sampler"), "dpmpp_2m")
        self._set_select("#g-scheduler", st_.get("scheduler"), "karras")
        for key, wid in (("steps", "#g-steps"), ("cfg", "#g-cfg"), ("seed", "#g-seed"), ("count", "#g-count")):
            self.query_one(wid, Input).value = str(st_.get(key, gen.DEFAULTS[key]))
        self.query_one("#g-pos", TextArea).text = st_.get("positive", "")
        self.query_one("#g-neg", TextArea).text = st_.get("negative", "")
        if missing:
            self.notify("Not found, left unchanged: " + ", ".join(os.path.basename(m) for m in missing),
                        severity="warning", title="Settings")

    def collect_settings(self):
        """Settings from the form, or raises ValueError naming the bad field."""
        def val(i):
            return self.query_one(i).value
        ckpt = val("#g-ckpt")
        if ckpt is Select.BLANK or not ckpt:
            raise ValueError("choose a checkpoint")
        loras = []
        for i in range(3):
            name = val(f"#g-lora{i}")
            if name not in (self.NONE, Select.BLANK, None):
                try:
                    loras.append({"name": name, "strength": float(val(f"#g-lstr{i}"))})
                except ValueError:
                    raise ValueError(f"LoRA {i + 1} strength must be a number") from None
        w, h = (int(x) for x in val("#g-size").split("x"))
        positive = self.query_one("#g-pos", TextArea).text.strip()
        if not positive:
            raise ValueError("write a prompt")
        try:
            steps, cfg = int(val("#g-steps")), float(val("#g-cfg"))
            seed, count = int(val("#g-seed") or -1), int(val("#g-count") or 1)
        except ValueError:
            raise ValueError("steps, CFG, seed and images must be numbers") from None
        if not (1 <= steps <= 150 and 1 <= count <= 100):
            raise ValueError("steps must be 1-150 and images 1-100")
        vae = val("#g-vae")
        return {"checkpoint": ckpt, "vae": "" if vae in (self.NONE, Select.BLANK) else vae, "loras": loras,
                "clip_skip": int(val("#g-clip")), "positive": positive,
                "negative": self.query_one("#g-neg", TextArea).text.strip(), "width": w, "height": h,
                "steps": steps, "cfg": cfg, "sampler": val("#g-sampler"), "scheduler": val("#g-scheduler"),
                "seed": seed, "count": count}

    def comfy_state(self):
        return (self.q.get("comfyui") or {}).get("state", "stopped")

    def ensure_comfy(self):
        """Asks the service to run ComfyUI; returns a note for the user."""
        if self.comfy_state() == "running":
            return ""
        with gq.locked() as q:
            q.setdefault("comfyui", {})["want"] = True
        cur = gq.current_job(self.q)
        return (f" ComfyUI starts after training job {cur['id']} finishes." if cur
                else " Starting ComfyUI first (about 30 s).")

    def queue_generation(self, settings, seeds_labels, group=None):
        """Submits (seed, label) images now if ComfyUI is up, else holds them
        until it is; held requests live in this app until then."""
        gen.save_last(settings)
        for seed, label, overrides in seeds_labels:
            self.gen_waiting.append((dict(settings, **overrides), seed, label, group))
        note = self.ensure_comfy()
        self.flush_generation()
        self.notify(f"{len(seeds_labels)} image(s) queued.{note}", title="Generate")

    def flush_generation(self):
        if not self.gen_waiting or self.comfy_state() != "running":
            return
        waiting, self.gen_waiting = self.gen_waiting, []
        for i, (st_, seed, label, group) in enumerate(waiting):
            try:
                gen.submit(st_, seed, label=label, group=group)
            except gen.ComfyError as e:
                self.gen_waiting = waiting[i:]
                self.notify(str(e), title="ComfyUI refused the request", severity="error", timeout=15)
                return

    def update_generate(self):
        self.flush_generation()
        c = self.q.get("comfyui") or {}
        state = c.get("state", "stopped")
        colors = {"running": "green", "starting": "yellow", "waiting": "yellow", "stopping": "yellow"}
        held = f"   [yellow]{len(self.gen_waiting)} request(s) waiting for ComfyUI[/]" if self.gen_waiting else ""
        prec = "fp32" if c.get("fp32", True) else "[yellow]fp16 (not validated on gfx803)[/]"
        self.query_one("#g-status", Label).update(
            f"ComfyUI [{colors.get(state, 'dim')}]● {state}[/]   precision {prec}"
            + (f"   [yellow]{c['note']}[/]" if c.get("note") else "") + held
            + ("" if state == "running" else "\n[dim]Queue starts ComfyUI if needed; training jobs pause meanwhile.[/dim]"))
        t = self.query_one("#g-queue-table", DataTable)
        keep = self.selected_key("#g-queue-table")
        t.clear()
        for st_, seed, label, _g in self.gen_waiting:
            t.add_row(label or st_["positive"][:30], "[yellow]waiting for ComfyUI[/]", "", key=f"held-{id(st_)}-{seed}")
        if state == "running":
            try:
                running, pending = gen.queue_state()
            except gen.ComfyError:
                running, pending = [], []
            for pid, meta in running:
                v, mx = self.gen_progress.get(pid, (0, 0))
                label = (meta or {}).get("label") or (meta or {}).get("positive", "")[:30] or "(web UI request)"
                t.add_row(label, "[b cyan]generating[/]", f"{v}/{mx}" if mx else "", key=pid)
            for pid, meta in pending:
                label = (meta or {}).get("label") or (meta or {}).get("positive", "")[:30] or "(web UI request)"
                t.add_row(label, "queued", "", key=pid)
        if keep:
            for i, k in enumerate(t.rows):
                if k.value == keep:
                    t.move_cursor(row=i)
        self.refresh_gallery()

    def refresh_gallery(self):
        items = gen.gallery()
        paths = [p for p, _ in items]
        if paths == [p for p, _ in self.gen_gallery]:
            return
        self.gen_gallery = items
        t = self.query_one("#g-gallery", DataTable)
        keep = self.selected_key("#g-gallery")
        t.clear()
        for p in paths:
            meta = self.gallery_meta(p)
            when = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
            label = (meta or {}).get("label") or (meta or {}).get("positive", os.path.basename(p))[:28]
            t.add_row(when, label, str((meta or {}).get("seed", "")), key=p)
        if keep in paths:
            t.move_cursor(row=paths.index(keep))
        self.show_gallery_item()

    def gallery_meta(self, path):
        if path not in self.gen_meta:
            self.gen_meta[path] = gen.image_settings(path)
        return self.gen_meta[path]

    @on(DataTable.RowHighlighted, "#g-gallery")
    def gallery_highlighted(self, _event):
        self.show_gallery_item()

    @on(DataTable.RowSelected, "#g-gallery")
    def gallery_selected(self, _event):
        self.gen_button("g-full")

    def show_gallery_item(self):
        path = self.selected_key("#g-gallery")
        prev = self.query_one("#g-prev")
        info = self.query_one("#g-prev-info", Label)
        if not path or not os.path.exists(path):
            prev.image = None
            info.update("[dim]Generated images appear here.[/dim]")
            return
        prev.image = path
        info.update(describe_generation(self.gallery_meta(path)))

    @work(exclusive=True, group="comfy-events")
    async def comfy_events(self):
        """Follows ComfyUI's websocket for per-step progress; reconnects while
        ComfyUI restarts or is stopped."""
        import aiohttp
        import asyncio
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(gen.COMFY_URL.replace("http", "ws") + "/ws?clientId=gfx803-app",
                                                  heartbeat=20) as ws:
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            data = msg.json()
                            d = data.get("data") or {}
                            if data.get("type") == "progress" and d.get("prompt_id"):
                                self.gen_progress[d["prompt_id"]] = (d.get("value", 0), d.get("max", 0))
                            elif data.get("type") == "executing" and d.get("node") is None and d.get("prompt_id"):
                                self.gen_progress.pop(d["prompt_id"], None)
                                self.refresh_gallery()
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError, ValueError):
                pass
            await asyncio.sleep(3)

    @work
    async def gen_button(self, bid):
        if bid == "g-queue":
            try:
                st_ = self.collect_settings()
            except ValueError as e:
                self.notify(str(e), severity="warning", title="Generate")
                return
            fixed = st_["seed"] >= 0
            items = [(st_["seed"] + i if fixed else gen.resolve_seed(-1), "", {}) for i in range(st_["count"])]
            self.queue_generation(st_, items)
        elif bid == "g-compare":
            try:
                st_ = self.collect_settings()
            except ValueError as e:
                self.notify(str(e), severity="warning", title="Generate")
                return
            opts = await self.push_screen_wait(CompareDialog(st_))
            if not opts:
                return
            labels = {name: lab for lab, name in self.lora_options}
            rest = st_["loras"][1:]
            items = []
            if opts["baseline"]:
                items.append((opts["seed"], "no LoRA", {"loras": rest}))
            for name in opts["names"]:
                lab = labels.get(name, os.path.basename(name)).replace(f"training: {opts['run']} ", "")
                items.append((opts["seed"], lab, {"loras": [{"name": name, "strength": opts["strength"]}] + rest}))
            self.queue_generation(dict(st_, count=1), items, group=gen.new_group())
        elif bid == "g-cancel":
            key = self.selected_key("#g-queue-table")
            if not key:
                return
            if key.startswith("held-"):
                self.gen_waiting = [w for w in self.gen_waiting if f"held-{id(w[0])}-{w[1]}" != key]
                return
            try:
                running, _ = gen.queue_state()
                gen.cancel(key, running=any(pid == key for pid, _ in running))
                self.notify("Cancelled.", title="Generate")
            except gen.ComfyError as e:
                self.notify(str(e), severity="error")
        elif bid in ("g-full", "g-use", "g-showcmp"):
            path = self.selected_key("#g-gallery")
            if not path:
                self.notify("Select an image in the gallery.", severity="warning")
                return
            meta = self.gallery_meta(path)
            if bid == "g-full":
                paths = [p for p, _ in self.gen_gallery]
                self.push_screen(ImageFull([(p, None) for p in paths], paths.index(path),
                                           describe=lambda p: describe_generation(self.gallery_meta(p))))
            elif bid == "g-use":
                if meta:
                    self.apply_settings(meta)
                    self.notify("Settings loaded into the form (seed included; set -1 for new ones).")
                else:
                    self.notify("This image has no gfx803 settings.", severity="warning")
            else:
                group = (meta or {}).get("group")
                if not group:
                    self.notify("This image is not part of a comparison.", severity="warning")
                    return
                items = [(p, self.gallery_meta(p)) for p, _ in self.gen_gallery
                         if (self.gallery_meta(p) or {}).get("group") == group]
                items.sort(key=lambda x: x[1].get("queued", 0))
                self.push_screen(CompareView(items))
        elif bid in ("g-load", "g-save", "g-saveas", "g-delete"):
            name = self.query_one("#g-preset", Select).value
            name = None if name is Select.BLANK else name
            if bid == "g-load":
                if name:
                    self.apply_settings(dict(gen.DEFAULTS, **gen.presets()[name]))
                    self.notify(f"Preset '{name}' loaded.")
                return
            if bid == "g-delete":
                if name and await self.push_screen_wait(Confirm(f"Delete preset '{name}'?", ok="Delete", danger=True)):
                    gen.delete_preset(name)
                    self.refresh_presets()
                return
            try:
                st_ = self.collect_settings()
            except ValueError as e:
                self.notify(str(e), severity="warning", title="Preset")
                return
            if bid == "g-save" and name:
                if not await self.push_screen_wait(Confirm(f"Replace preset '{name}' with the form's settings?",
                                                           ok="Replace")):
                    return
            else:
                name = await self.push_screen_wait(TextDialog("Save preset", "Name for these settings:",
                                                              placeholder="e.g. portrait, 512x768", ok="Save"))
                name = (name or "").strip()
                if not name:
                    return
                if name in gen.presets() and not await self.push_screen_wait(
                        Confirm(f"A preset named '{name}' exists. Replace it?", ok="Replace")):
                    return
            gen.save_preset(name, st_)
            self.refresh_presets(select=name)
            self.notify(f"Preset '{name}' saved.")

    # ---- actions
    def action_tab(self, tab):
        # Drop focus first: TabbedContent shows whichever pane holds the
        # focused widget, so a table still focused in the old tab would pull
        # the view straight back.
        self.set_focus(None)
        self.query_one(TabbedContent).active = tab

    def action_add_run(self):
        self.push_screen(AddRun())

    def action_toggle_comfy(self):
        with gq.locked() as q:
            c = q.setdefault("comfyui", {})
            c["want"] = not c.get("want")
            want = c["want"]
        cur = gq.current_job(self.q)
        if want and cur:
            self.notify(f"ComfyUI will start when job {cur['id']} finishes; no new jobs start meanwhile.",
                        title="ComfyUI")
        else:
            self.notify("Starting ComfyUI..." if want else "Stopping ComfyUI...", title="ComfyUI")

    def action_toggle_pause(self):
        with gq.locked() as q:
            q["paused"] = not q["paused"]
            q["pause_reason"] = "paused by user" if q["paused"] else ""
            if not q["paused"]:
                q["consecutive_failures"] = 0

    @work
    async def action_shutdown(self):
        cur = gq.current_job(self.q)
        msg = ("Shut down the service? ComfyUI stops"
               + (f", and training job {cur['id']} is interrupted; it resumes from its last checkpoint "
                  "the next time the service starts." if cur else "."))
        if await self.push_screen_wait(Confirm(msg, ok="Shut down", danger=True)):
            with gq.locked() as q:
                q["shutdown"] = True
            self.notify("Service shutting down; the container stops when it is done.", title="gfx803")

    @work
    async def request_job(self, what):
        cur = gq.current_job(self.q)
        if not cur:
            self.notify("No job is training.", severity="warning")
            return
        msg = (f"Stop job {cur['id']} ({cur['label']}) and pause the queue?" if what == "stop"
               else f"Skip job {cur['id']} ({cur['label']}) and go on with the next one?")
        msg += "\nThe run keeps its checkpoints and can be retried later; steps since the last checkpoint are lost."
        if await self.push_screen_wait(Confirm(msg, ok="Stop" if what == "stop" else "Skip", danger=True)):
            with gq.locked() as q:
                j = gq.find(q, cur["id"])
                if j and j["state"] == gq.RUNNING:
                    j["request"] = what

    def queue_job(self):
        key = self.selected_key("#queue-table")
        return gq.find(self.q, int(key)) if key else None

    @on(Button.Pressed)
    def buttons(self, event):
        bid = event.button.id
        if bid in ("ov-add", "q-add"):
            self.action_add_run()
        elif bid in ("ov-pause", "q-pause"):
            self.action_toggle_pause()
        elif bid in ("ov-stop", "q-stop"):
            self.request_job("stop")
        elif bid == "q-skip":
            self.request_job("skip")
        elif bid in ("ov-comfy", "c-toggle"):
            self.action_toggle_comfy()
        elif bid in ("q-up", "q-down", "q-remove", "q-retry"):
            job = self.queue_job()
            if not job:
                return
            with gq.locked() as q:
                j = gq.find(q, job["id"])
                if not j:
                    return
                i = q["jobs"].index(j)
                if bid in ("q-up", "q-down"):
                    k = i - 1 if bid == "q-up" else i + 1
                    if 0 <= k < len(q["jobs"]):
                        q["jobs"][i], q["jobs"][k] = q["jobs"][k], q["jobs"][i]
                elif bid == "q-remove":
                    if j["state"] == gq.RUNNING:
                        self.notify("The job is training; stop or skip it first.", severity="warning")
                    else:
                        q["jobs"].remove(j)
                elif bid == "q-retry":
                    if j["state"] in (gq.FAILED, gq.STOPPED, gq.SKIPPED):
                        if j["run"] and gq.run_state(j["run"]) not in ("completed", ""):
                            j["args"] = ["--resume-run", j["run"]]
                        j.update(state=gq.PENDING, note="retried", request=None)
                    else:
                        self.notify("Only failed, stopped or skipped jobs can be retried.", severity="warning")
            self.update_queue_table()
            if bid in ("q-up", "q-down"):
                t = self.query_one("#queue-table", DataTable)
                t.move_cursor(row=max(min(t.cursor_row + (-1 if bid == "q-up" else 1), t.row_count - 1), 0))
        elif bid.startswith("g-"):
            self.gen_button(bid)
        elif bid.startswith("d-"):
            self.dataset_action(bid[2:])
        elif bid in ("r-log", "r-copy", "r-resume"):
            run = self.selected_key("#runs-table")
            if not run:
                self.notify("Select a run first.", severity="warning")
                return
            if bid == "r-log":
                self.push_screen(LogView(f"Training log: {run}", os.path.join(gq.RUNS_DIR, run, "train.log")))
            elif bid == "r-copy":
                self.push_screen(PickCheckpoint(run))
            else:
                state = gq.run_state(run)
                if state in ("completed", "running"):
                    self.notify(f"Run is {state}; nothing to resume.", severity="warning")
                else:
                    job = gq.add_job(["--resume-run", run])
                    self.notify(f"Job {job} queued to resume {run}.", title="Training queue")

    @on(Input.Submitted, "#q-cooldown")
    def set_cooldown(self, event):
        if event.value.isdigit():
            with gq.locked() as q:
                q["cooldown_s"] = int(event.value)
            self.notify(f"Cool-down between jobs: {event.value} s")

    @on(Switch.Changed, "#c-fp32")
    def set_fp32(self, event):
        with gq.locked() as q:
            c = q.setdefault("comfyui", {})
            if c.get("fp32") != event.value:
                c["fp32"] = event.value
                if c.get("state") == "running":
                    self.notify("Applies the next time ComfyUI starts.", title="ComfyUI precision")


def main():
    Gfx803App().run()


if __name__ == "__main__":
    main()
