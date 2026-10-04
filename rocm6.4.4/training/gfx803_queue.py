#!/usr/bin/env python3
"""Training queue for gfx803-train: run LoRA trainings one after another.

  gfx803-queue                       show the queue (gfx803-app is the interactive view)
  gfx803-queue runner                run queued jobs without the service (no ComfyUI control)
  gfx803-queue add [--label TEXT] -- <gfx803-train args>
  gfx803-queue list
  gfx803-queue pause | resume        hold the queue after the current job / let it continue
  gfx803-queue stop | skip           stop the current job and pause / stop it and go on
  gfx803-queue remove ID | retry ID  drop a pending job / requeue a failed or stopped one
  gfx803-queue cooldown SECONDS      wait between jobs

The runner (inside gfx803-service, or standalone) and every viewer share
$WORKSPACE/queue/queue.json, so viewers can be closed and reopened from any
terminal while jobs keep running. Each job is an ordinary gfx803-train run
in its own process, with its own run record under $WORKSPACE/runs/.

Failure policy: a failed job is marked and the queue moves on; two failures in
a row pause it, since that usually means the GPU or driver is in a bad state
(after a hang and reset, every later job fails the same way).
"""
import contextlib
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
QUEUE_DIR = os.path.join(WORKSPACE, "queue")
QUEUE_FILE = os.path.join(QUEUE_DIR, "queue.json")
LOCK_FILE = os.path.join(QUEUE_DIR, "queue.lock")
RUNNER_LOCK = os.path.join(QUEUE_DIR, "runner.lock")
LOG_DIR = os.path.join(QUEUE_DIR, "logs")
RUNS_DIR = os.path.join(WORKSPACE, "runs")
TRAIN_CMD = os.environ.get("GFX803_QUEUE_TRAIN_CMD", "gfx803-train").split()
MAX_CONSECUTIVE_FAILURES = 2
# A job interrupted by a container restart or reboot is resumed this many
# times before it counts as failed, so a run that dies at the same point every
# time cannot hold up the queue forever.
MAX_AUTO_RESUMES = 2
POLL_S = 1.0

PENDING, RUNNING, DONE, FAILED, STOPPED, SKIPPED = (
    "pending", "running", "done", "failed", "stopped", "skipped")


# ------------------------------------------------------------------ store
def empty_queue():
    return {"paused": False, "pause_reason": "", "cooldown_s": 0, "next_id": 1,
            "consecutive_failures": 0, "jobs": []}


@contextlib.contextmanager
def locked():
    """Yields the queue under an exclusive lock and saves it on exit.

    The dashboard and the runner both write the file; the lock serializes
    them and the rename makes every save all-or-nothing.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOCK_FILE, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        q = load()
        yield q
        tmp = QUEUE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(q, f, indent=1)
        os.replace(tmp, QUEUE_FILE)


def load():
    try:
        with open(QUEUE_FILE) as f:
            q = json.load(f)
    except (OSError, ValueError):
        return empty_queue()
    base = empty_queue()
    base.update(q)
    return base


def find(q, job_id):
    for j in q["jobs"]:
        if j["id"] == job_id:
            return j
    return None


def add_job(args, label=""):
    with locked() as q:
        job = {"id": q["next_id"], "label": label or describe_args(args), "args": args,
               "state": PENDING, "run": run_name_from_args(args), "pid": None,
               "started": None, "ended": None, "rc": None, "note": "", "resumes": 0,
               "request": None, "added": time.time()}
        q["next_id"] += 1
        q["jobs"].append(job)
        return job["id"]


def describe_args(args):
    def opt(name):
        return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None
    if opt("--resume-run"):
        return "resume " + opt("--resume-run")
    parts = [opt("--data") or "?", os.path.splitext(os.path.basename(opt("--model") or "default model"))[0]]
    if opt("--steps"):
        parts.append(opt("--steps") + " steps")
    return " / ".join(parts)


def run_name_from_args(args):
    for flag in ("--resume-run", "--name", "-n"):
        if flag in args and args.index(flag) + 1 < len(args):
            return args[args.index(flag) + 1]
    return None


def read_kv(path):
    kv = {}
    try:
        with open(path) as f:
            for line in f:
                k, sep, v = line.rstrip("\n").partition("=")
                if sep:
                    kv[k] = v
    except OSError:
        pass
    return kv


def run_state(run):
    return read_kv(os.path.join(RUNS_DIR, run, "status")).get("STATE", "") if run else ""


def runner_alive():
    """True if a runner holds the runner lock."""
    os.makedirs(QUEUE_DIR, exist_ok=True)
    with open(RUNNER_LOCK, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def process_is_job(pid):
    """True if pid is still the job's training wrapper in this container."""
    if not pid:
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ")
    except OSError:
        return False
    return b"gfx803-train" in cmd or TRAIN_CMD[-1].encode() in cmd


# ------------------------------------------------------------------ runner
class Runner:
    def __init__(self):
        self.proc = None
        self.job_id = None

    def log(self, msg):
        print(time.strftime("%H:%M:%S ") + msg, flush=True)

    def start(self, job, args):
        logf = open(os.path.join(LOG_DIR, f"{job['id']}.log"), "ab")
        env = dict(os.environ, GFX803_TRAIN_NO_MENU="1")
        # Its own session, so a stop request can signal the wrapper alone
        # (it then stops its training process group itself), and so the job
        # outlives a runner that is killed: a restarted runner reattaches.
        self.proc = subprocess.Popen(TRAIN_CMD + args, stdin=subprocess.DEVNULL, stdout=logf,
                                     stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.job_id = job["id"]
        job.update(state=RUNNING, pid=self.proc.pid, started=time.time(), ended=None, rc=None,
                   request=None)
        self.log(f"job {job['id']} started: {job['label']} (pid {self.proc.pid})")

    def finish(self, q, job, rc):
        state = run_state(job["run"])
        req = job.get("request")
        if req in ("stop", "skip"):
            job["state"] = STOPPED if req == "stop" else SKIPPED
            if req == "stop":
                q["paused"], q["pause_reason"] = True, f"job {job['id']} stopped"
        elif state == "completed" or (rc == 0 and "--dry-run" in job["args"]):
            job["state"] = DONE
        else:
            job["state"] = FAILED
            job["note"] = f"run state {state or 'unknown'}, exit {rc}"
        job.update(ended=time.time(), rc=rc, pid=None, request=None)
        if job["state"] == FAILED:
            q["consecutive_failures"] += 1
            if q["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
                q["paused"] = True
                q["pause_reason"] = (f"{q['consecutive_failures']} jobs failed in a row; check the GPU "
                                     "(dmesg, gpu_monitor.log) before resuming")
        elif job["state"] == DONE:
            q["consecutive_failures"] = 0
        self.log(f"job {job['id']} {job['state']}" + (f": {job['note']}" if job["note"] else ""))
        self.proc, self.job_id = None, None

    def learn_run_name(self, job):
        """Reads the run name from the wrapper's 'Starting: run=NAME' line."""
        if job["run"]:
            return
        try:
            with open(os.path.join(LOG_DIR, f"{job['id']}.log"), "rb") as f:
                m = re.search(rb"run=(\S+)", f.read())
        except OSError:
            m = None
        if m:
            job["run"] = m.group(1).decode()

    def recover(self, q):
        """Handles jobs left running by a previous runner."""
        for job in q["jobs"]:
            if job["state"] != RUNNING:
                continue
            if process_is_job(job["pid"]):
                self.log(f"job {job['id']} still running (pid {job['pid']}); watching it")
                self.job_id = job["id"]
                continue
            self.learn_run_name(job)
            if run_state(job["run"]) == "completed":
                job.update(state=DONE, pid=None, ended=time.time())
            elif job["run"] and job["resumes"] < MAX_AUTO_RESUMES:
                job["resumes"] += 1
                job.update(state=PENDING, pid=None, note=f"auto-resume {job['resumes']} after restart")
                if "--resume-run" not in job["args"]:
                    job["args"] = ["--resume-run", job["run"]]
                self.log(f"job {job['id']} was interrupted; queued to resume {job['run']}")
            else:
                job.update(state=FAILED, pid=None, ended=time.time(),
                           note="interrupted by a restart" + (" too often" if job["run"] else " before it started"))

    def acquire(self):
        """Takes the runner lock for the life of the process; exits if held."""
        os.makedirs(LOG_DIR, exist_ok=True)
        self.lockf = open(RUNNER_LOCK, "a")
        try:
            fcntl.flock(self.lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit("gfx803-queue: a runner is already running")
        with locked() as q:
            self.recover(q)
        self.log(f"runner started (queue {QUEUE_FILE})")
        self.cooldown_until = 0

    def tick(self, q, can_start=True):
        """One pass over the queue, under the caller's lock.

        can_start=False holds back new jobs (the service sets it while
        ComfyUI has the GPU); a job already training is never affected.
        """
        job = find(q, self.job_id) if self.job_id else None
        if job:
            self.learn_run_name(job)
            if job.get("request") in ("stop", "skip") and not job.get("signalled"):
                job["signalled"] = True
                self.log(f"job {job['id']}: {job['request']} requested; interrupting it")
                with contextlib.suppress(ProcessLookupError, TypeError):
                    os.kill(job["pid"], signal.SIGINT)
            rc = None
            if self.proc is not None:
                rc = self.proc.poll()
                done = rc is not None
            else:
                done = not process_is_job(job["pid"])
            if done:
                job.pop("signalled", None)
                self.finish(q, job, rc)
                self.cooldown_until = time.time() + q["cooldown_s"]
        elif can_start and not q["paused"] and time.time() >= self.cooldown_until:
            nxt = next((j for j in q["jobs"] if j["state"] == PENDING), None)
            if nxt:
                self.start(nxt, nxt["args"])

    def interrupt_current(self, timeout=40):
        """Interrupts the training job so it can resume later; used when the
        container is stopped. The job stays RUNNING, so the next start
        resumes it from its last checkpoint."""
        if not self.job_id:
            return
        with locked() as q:
            job = find(q, self.job_id)
            pid = job["pid"] if job else None
        if not pid:
            return
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGINT)
        end = time.time() + timeout
        while time.time() < end and process_is_job(pid):
            time.sleep(0.5)

    def loop(self):
        self.acquire()
        while True:
            with locked() as q:
                self.tick(q)
            time.sleep(POLL_S)


# ------------------------------------------------------------------ CLI
def current_job(q):
    return next((j for j in q["jobs"] if j["state"] == RUNNING), None)


def cmd_list(_):
    q = load()
    print(f"runner: {'running' if runner_alive() else 'NOT running'}   "
          f"queue: {'PAUSED (' + q['pause_reason'] + ')' if q['paused'] else 'active'}   "
          f"cooldown {q['cooldown_s']} s")
    if not q["jobs"]:
        print("no jobs")
    for j in q["jobs"]:
        print(f"{j['id']:4d}  {j['state']:8s}  {j['label']}" + (f"   [{j['note']}]" if j["note"] else ""))
        print(f"        gfx803-train {' '.join(map(shlex_quote, j['args']))}")


def shlex_quote(s):
    import shlex
    return shlex.quote(s)


def main(argv):
    if not argv:
        cmd_list(None)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "runner":
        Runner().loop()
    elif cmd == "add":
        label = ""
        if rest[:1] == ["--label"]:
            label, rest = rest[1], rest[2:]
        if rest[:1] == ["--"]:
            rest = rest[1:]
        if not rest:
            sys.exit("usage: gfx803-queue add [--label TEXT] -- <gfx803-train args>")
        if not any(a in rest for a in ("--resume", "--resume-run")):
            rest = rest + ["--resume", "never"] if "--" not in rest else (
                rest[:rest.index("--")] + ["--resume", "never"] + rest[rest.index("--"):])
        print(f"added job {add_job(rest, label)}")
    elif cmd == "list":
        cmd_list(None)
    elif cmd in ("pause", "resume"):
        with locked() as q:
            q["paused"] = cmd == "pause"
            q["pause_reason"] = "paused by user" if cmd == "pause" else ""
            if cmd == "resume":
                q["consecutive_failures"] = 0
    elif cmd in ("stop", "skip"):
        with locked() as q:
            job = current_job(q)
            if not job:
                sys.exit("no job is running")
            job["request"] = cmd
    elif cmd in ("remove", "retry"):
        job_id = int(rest[0])
        with locked() as q:
            job = find(q, job_id)
            if not job:
                sys.exit(f"no job {job_id}")
            if cmd == "remove":
                if job["state"] == RUNNING:
                    sys.exit("job is running; stop or skip it first")
                q["jobs"].remove(job)
            else:
                if job["state"] not in (FAILED, STOPPED, SKIPPED):
                    sys.exit(f"job {job_id} is {job['state']}")
                if job["run"] and run_state(job["run"]) not in ("completed", ""):
                    job["args"] = ["--resume-run", job["run"]]
                job.update(state=PENDING, note="retried", request=None)
    elif cmd == "cooldown":
        with locked() as q:
            q["cooldown_s"] = max(int(rest[0]), 0)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
