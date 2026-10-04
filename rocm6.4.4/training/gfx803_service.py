#!/usr/bin/env python3
"""gfx803-service: the gfx803 container's main process.

Runs the LoRA training queue and the ComfyUI server, one GPU user at a time,
and keeps both going while gfx803-app (the terminal UI) comes and goes. All
state lives in $WORKSPACE/queue/queue.json, which the app edits under the
same lock.

One GPU user at a time: an 8 GB RX 590 cannot hold SD1.5 training and a
loaded ComfyUI model together, and when VRAM fills, the driver moves buffers
to system memory and both slow to a crawl. So while ComfyUI is wanted or
running, no new training job starts, and a ComfyUI start waits for the
training job that is running to finish.

On SIGTERM (docker stop, host shutdown) the training job is interrupted the
way Ctrl+C would, so it resumes from its last checkpoint on the next start,
and ComfyUI is stopped.
"""
import contextlib
import os
import signal
import subprocess
import sys
import time
import urllib.request

import gfx803_queue as gq

COMFYUI_DIR = "/opt/ComfyUI"
COMFYUI_PORT = 8188
COMFYUI_LOG = os.path.join(gq.QUEUE_DIR, "comfyui.log")
# ComfyUI has no authentication, and ComfyUI-Manager installs and runs code, so
# it accepts connections from this machine only unless GFX803_COMFYUI_LISTEN
# names another address (0.0.0.0 for every interface, e.g. to use it from
# another device on the LAN). The container shares the host's network, so the
# address is the host's. The other arguments match the image's default command,
# so ComfyUI behaves the same whether it is started here or by start-comfyui.sh.
COMFYUI_LISTEN = os.environ.get("GFX803_COMFYUI_LISTEN") or "127.0.0.1"
COMFYUI_ARGS = ["--listen", COMFYUI_LISTEN, "--port", str(COMFYUI_PORT), "--lowvram", "--use-split-cross-attention"]
COMFYUI_STOP_TIMEOUT_S = 20


def comfy_defaults():
    # fp32 by default: ComfyUI's fp16 on ROCm is not validated on gfx803, and
    # fp32 is the precision everything else here was checked in.
    return {"want": False, "fp32": True, "state": "stopped", "pid": None, "started": None, "note": ""}


def comfy_http_ok():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{COMFYUI_PORT}/system_stats", timeout=1) as r:
            return r.status == 200
    except OSError:
        return False


def pid_is_comfyui(pid):
    if not pid:
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"main.py" in f.read()
    except OSError:
        return False


class ComfyUI:
    def __init__(self, log):
        self.log = log
        self.proc = None
        self.stop_sent = None

    def start(self, c):
        os.makedirs(gq.QUEUE_DIR, exist_ok=True)
        logf = open(COMFYUI_LOG, "ab")
        logf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} starting ComfyUI"
                   f"{' (fp32)' if c['fp32'] else ''}\n".encode())
        logf.flush()
        import gfx803_generate as gen
        args = COMFYUI_ARGS + ["--extra-model-paths-config", gen.extra_model_paths_yaml()] \
            + (["--force-fp32"] if c["fp32"] else [])
        # The image's default command copies bundled custom nodes into the
        # mounted custom_nodes folder first; so does this.
        cmd = ("cp -an /opt/staging_nodes/* /opt/ComfyUI/custom_nodes/ 2>/dev/null || true; "
               "exec python3 main.py " + " ".join(args))
        self.proc = subprocess.Popen(["sh", "-c", cmd], cwd=COMFYUI_DIR, stdin=subprocess.DEVNULL,
                                     stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        c.update(state="starting", pid=self.proc.pid, started=time.time(), note="")
        self.stop_sent = None
        self.log(f"ComfyUI starting (pid {self.proc.pid})")

    def alive(self, c):
        if self.proc is not None:
            return self.proc.poll() is None
        return pid_is_comfyui(c["pid"])

    def stop(self, c):
        if self.stop_sent is None:
            self.stop_sent = time.time()
            c["state"] = "stopping"
            self.log("ComfyUI stopping")
            with contextlib.suppress(ProcessLookupError, TypeError):
                os.killpg(c["pid"], signal.SIGTERM)
        elif time.time() - self.stop_sent > COMFYUI_STOP_TIMEOUT_S:
            with contextlib.suppress(ProcessLookupError, TypeError):
                os.killpg(c["pid"], signal.SIGKILL)

    def tick(self, q, training_active):
        c = q.setdefault("comfyui", comfy_defaults())
        for k, v in comfy_defaults().items():
            c.setdefault(k, v)
        alive = self.alive(c)
        if alive:
            if not c["want"]:
                self.stop(c)
            elif c["state"] in ("starting", "stopped", "waiting") and comfy_http_ok():
                c["state"] = "running"
                self.log("ComfyUI ready")
        else:
            if c["state"] in ("starting", "running", "stopping"):
                rc = self.proc.poll() if self.proc is not None else None
                expected = c["state"] == "stopping" or not c["want"]
                c.update(state="stopped", pid=None,
                         note="" if expected else f"exited unexpectedly (code {rc}); see queue/comfyui.log")
                if not expected:
                    c["want"] = False
                    self.log(f"ComfyUI exited unexpectedly (code {rc})")
                self.proc, self.stop_sent = None, None
            if c["want"]:
                if training_active:
                    c["state"], c["note"] = "waiting", "starts when the training job finishes"
                else:
                    self.start(c)

    def shutdown(self):
        q = gq.load()
        c = q.get("comfyui", comfy_defaults())
        if self.alive(c):
            with contextlib.suppress(ProcessLookupError, TypeError):
                os.killpg(c["pid"], signal.SIGTERM)
            end = time.time() + COMFYUI_STOP_TIMEOUT_S
            while time.time() < end and self.alive(c):
                time.sleep(0.5)
        with gq.locked() as q:
            q.setdefault("comfyui", comfy_defaults()).update(state="stopped", pid=None)


def main():
    runner = gq.Runner()
    comfy = ComfyUI(runner.log)
    stopping = {"flag": False}

    def on_term(_sig, _frame):
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    runner.acquire()
    with gq.locked() as q:
        q["shutdown"] = False
        c = q.setdefault("comfyui", comfy_defaults())
        if not pid_is_comfyui(c.get("pid")):
            c.update(state="stopped", pid=None)
        q["service"] = {"pid": os.getpid(), "started": time.time()}
    runner.log("service started")

    while not stopping["flag"]:
        with gq.locked() as q:
            if q.get("shutdown"):
                stopping["flag"] = True
                break
            training_active = gq.current_job(q) is not None
            comfy.tick(q, training_active)
            c = q["comfyui"]
            gpu_for_comfy = c["want"] or c["state"] != "stopped"
            runner.tick(q, can_start=not gpu_for_comfy)
            q["service"]["heartbeat"] = time.time()
        time.sleep(gq.POLL_S)

    runner.log("service stopping: interrupting training and stopping ComfyUI")
    runner.interrupt_current()
    comfy.shutdown()
    with gq.locked() as q:
        q["service"] = {}
        q["shutdown"] = False
    runner.log("service stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
