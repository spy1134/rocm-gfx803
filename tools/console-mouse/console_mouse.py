#!/usr/bin/env python3
"""Give a terminal program the mouse on a Linux text console, through gpm.

    console_mouse.py [--log FILE] [--] COMMAND [ARGS...]

Terminal UIs (Textual, curses with xterm mouse support, editors) read the mouse
as xterm escape sequences that a terminal emulator writes into their input. The
Linux console never writes those. gpm reads the mouse there, but only hands
events to programs linked against libgpm that run directly on the console, and
libgpm refuses a program behind a pseudo-terminal: one inside screen or tmux,
or one reached through `docker exec -it`.

This wrapper runs COMMAND on a pseudo-terminal of its own and passes the
keyboard and the screen through unchanged. While COMMAND has xterm mouse
reporting switched on (DECSET 9, 1000, 1002 or 1003, seen in its output), the
wrapper connects to gpm's socket for the active virtual console and writes each
click, drag, release and wheel step into COMMAND's input in the encoding it
asked for (SGR with DECSET 1006, the legacy X10 bytes otherwise). While
reporting is off it stays disconnected, so gpm's own copy and paste works as
usual.

It does nothing extra, and simply runs COMMAND, when stdin is not a terminal,
when gpm is not running, or when the active virtual console is in graphics
mode (a desktop session, whose terminal emulator already sends the mouse
itself). Reattaching a screen session from another place is handled: the
console is re-checked every second.

gpm reports positions on the whole console, and they are passed on as they are,
so they line up only while COMMAND fills the console: one full-size window in
screen, no status or caption line.

--log FILE appends every gpm event and every sequence sent, for checking a
mouse's wheel direction and button mapping.
"""

import errno
import fcntl
import os
import re
import select
import signal
import socket
import struct
import sys
import termios
import time
import tty

GPM_SOCKET = os.environ.get("CONSOLE_MOUSE_GPM_SOCKET", "/dev/gpmctl")

# gpm.h: struct Gpm_Connect and struct Gpm_Event, native layout.
CONNECT = struct.Struct("=HHHHii")
EVENT = struct.Struct("=BBHhhhhiiihh")

GPM_MOVE, GPM_DRAG, GPM_DOWN, GPM_UP = 1, 2, 4, 8
GPM_HARD = 256
GPM_B_RIGHT, GPM_B_MIDDLE, GPM_B_LEFT = 1, 2, 4
GPM_B_UP, GPM_B_DOWN = 16, 32

# linux/keyboard.h shift-state bits, as gpm reports them in `modifiers`.
KG_SHIFT, KG_CTRL, KG_ALT, KG_ALTGR = 1 << 0, 1 << 2, 1 << 3, 1 << 1

KDGETMODE = 0x4B3B
KD_TEXT = 0

MOUSE_MODES = {9, 1000, 1002, 1003}
DECSET = re.compile(rb"\x1b\[\?([0-9;]+)([hl])")
INCOMPLETE_CSI = re.compile(rb"\x1b(\[[\x30-\x3f]*[\x20-\x2f]*)?\Z")


def active_console():
    """The active virtual console's number, if it is a text console we own."""
    try:
        with open("/sys/class/tty/tty0/active") as f:
            name = f.read().strip()
        vc = int(name[3:]) if name.startswith("tty") else 0
        path = "/dev/" + name
        # gpm accepts a client for a console only from that console's owner.
        if vc <= 0 or os.stat(path).st_uid != os.getuid():
            return None
        fd = os.open(path, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            mode = struct.unpack("i", fcntl.ioctl(fd, KDGETMODE, b"\0" * 4))[0]
        finally:
            os.close(fd)
        return vc if mode == KD_TEXT else None
    except (OSError, ValueError):
        return None


def connect_gpm(vc):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(GPM_SOCKET)
        # The pointer is drawn by gpm's default handler, so motion also goes
        # there (GPM_HARD passes an event on after this client has it). Clicks
        # and drags do not, or gpm would start a text selection under the UI.
        s.sendall(CONNECT.pack(GPM_MOVE | GPM_DRAG | GPM_DOWN | GPM_UP,
                               GPM_MOVE | GPM_HARD, 0, 0xFFFF, os.getpid(), vc))
    except OSError:
        s.close()
        return None
    return s


class MouseEncoder:
    """Turns gpm events into the xterm reports the program asked for."""

    def __init__(self):
        self.modes = set()
        self.sgr = False
        self.held = 0

    def track(self, data, tail):
        buf = tail + data
        for m in DECSET.finditer(buf):
            on = m.group(2) == b"h"
            for p in m.group(1).split(b";"):
                if not p:
                    continue
                n = int(p)
                if n in MOUSE_MODES:
                    (self.modes.add if on else self.modes.discard)(n)
                elif n == 1006:
                    self.sgr = on
        if b"\x1bc" in buf:  # RIS resets every mode
            self.modes.clear()
            self.sgr = False
        last = buf.rfind(b"\x1b")
        if last >= 0 and len(buf) - last < 32 and INCOMPLETE_CSI.match(buf, last):
            return buf[last:]
        return b""

    @property
    def wanted(self):
        return bool(self.modes)

    def _report(self, code, x, y, release=False):
        if self.sgr:
            return b"\x1b[<%d;%d;%d%c" % (code, x, y, ord("m" if release else "M"))
        if release:
            code = (code & ~3) | 3
        if x > 223 or y > 223:
            return b""
        return b"\x1b[M" + bytes((32 + code, 32 + x, 32 + y))

    @staticmethod
    def _button(bits):
        for bit, code in ((GPM_B_LEFT, 0), (GPM_B_MIDDLE, 1), (GPM_B_RIGHT, 2)):
            if bits & bit:
                return code
        return None

    def encode(self, ev, cols, rows):
        buttons, mods, _vc, _dx, _dy, x, y, etype, _clicks, _margin, _wdx, wdy = ev
        x = min(max(x, 1), cols)
        y = min(max(y, 1), rows)
        modbits = ((4 if mods & KG_SHIFT else 0) | (8 if mods & (KG_ALT | KG_ALTGR) else 0)
                   | (16 if mods & KG_CTRL else 0))
        bare = etype & 0x0F
        out = b""

        # Wheels arrive as a vertical wheel delta, or on some mouse protocols
        # as presses of the extra GPM_B_UP/GPM_B_DOWN buttons.
        steps = []
        if wdy:
            steps = [64 if wdy > 0 else 65] * min(abs(wdy), 10)
        elif bare == GPM_DOWN and buttons & (GPM_B_UP | GPM_B_DOWN):
            steps = [64 if buttons & GPM_B_UP else 65]
        if steps:
            return b"".join(self._report(c | modbits, x, y) for c in steps)

        buttons &= GPM_B_LEFT | GPM_B_MIDDLE | GPM_B_RIGHT
        if bare == GPM_DOWN:
            pressed = buttons & ~self.held
            self.held |= buttons
            for bit in (GPM_B_LEFT, GPM_B_MIDDLE, GPM_B_RIGHT):
                if pressed & bit:
                    out += self._report(self._button(bit) | modbits, x, y)
        elif bare == GPM_UP:
            released = buttons & self.held if buttons else self.held
            self.held &= ~released
            if self.modes - {9}:
                for bit in (GPM_B_LEFT, GPM_B_MIDDLE, GPM_B_RIGHT):
                    if released & bit:
                        out += self._report(self._button(bit) | modbits, x, y, release=True)
        elif bare == GPM_DRAG and self.modes & {1002, 1003}:
            code = self._button(buttons or self.held)
            out = self._report((3 if code is None else code) | 32 | modbits, x, y)
        elif bare == GPM_MOVE and 1003 in self.modes:
            self.held = 0
            out = self._report(3 | 32 | modbits, x, y)
        return out


def window_size(fd):
    return fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8)


def main(argv):
    log = None
    if len(argv) >= 2 and argv[0] == "--log":
        log = open(argv[1], "a", buffering=1)
        argv = argv[2:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        sys.exit("usage: console_mouse.py [--log FILE] [--] COMMAND [ARGS...]")

    if not (os.isatty(0) and os.isatty(1) and os.path.exists(GPM_SOCKET)):
        os.execvp(argv[0], argv)

    size = window_size(0)
    pid, master = os.forkpty()
    if pid == 0:
        try:
            os.execvp(argv[0], argv)
        except OSError as e:
            print("console_mouse: %s: %s" % (argv[0], e.strerror), file=sys.stderr)
        os._exit(127)

    fcntl.ioctl(master, termios.TIOCSWINSZ, size)

    def resize(*_):
        try:
            fcntl.ioctl(master, termios.TIOCSWINSZ, window_size(0))
            os.kill(pid, signal.SIGWINCH)
        except OSError:
            pass

    def hangup(signum, _frame):
        try:
            os.kill(pid, signum)
        except OSError:
            pass

    signal.signal(signal.SIGWINCH, resize)
    signal.signal(signal.SIGHUP, hangup)
    signal.signal(signal.SIGTERM, hangup)

    saved = termios.tcgetattr(0)
    tty.setraw(0)
    enc = MouseEncoder()
    tail = b""
    gpm, gpm_vc, gpm_buf = None, None, b""
    next_check = 0.0
    stdin_open = True
    try:
        while True:
            now = time.monotonic()
            if now >= next_check:
                vc = active_console() if enc.wanted else None
                if gpm is not None and vc != gpm_vc:
                    gpm.close()
                    gpm, gpm_buf = None, b""
                if gpm is None and vc is not None:
                    gpm, gpm_vc = connect_gpm(vc), vc
                next_check = now + 1.0

            fds = [master] + ([0] if stdin_open else []) + ([gpm] if gpm else [])
            try:
                ready, _, _ = select.select(fds, [], [], 1.0)
            except InterruptedError:
                continue

            if master in ready:
                try:
                    data = os.read(master, 65536)
                except OSError as e:
                    if e.errno != errno.EIO:
                        raise
                    data = b""
                if not data:
                    break
                os.write(1, data)
                was = enc.wanted
                tail = enc.track(data, tail)
                if enc.wanted != was:
                    next_check = 0.0

            if 0 in ready:
                data = os.read(0, 65536)
                if data:
                    os.write(master, data)
                else:
                    stdin_open = False

            if gpm is not None and gpm in ready:
                chunk = gpm.recv(4096)
                if not chunk:
                    gpm.close()
                    gpm, gpm_buf = None, b""
                    continue
                gpm_buf += chunk
                rows, cols = struct.unpack("HHHH", window_size(master))[:2]
                while len(gpm_buf) >= EVENT.size:
                    ev = EVENT.unpack_from(gpm_buf)
                    gpm_buf = gpm_buf[EVENT.size:]
                    out = enc.encode(ev, cols, rows) if enc.wanted else b""
                    if out:
                        os.write(master, out)
                    if log:
                        log.write("%.3f %r -> %r\n" % (time.time(), ev, out))
    finally:
        termios.tcsetattr(0, termios.TCSAFLUSH, saved)
        if gpm is not None:
            gpm.close()

    _, status = os.waitpid(pid, 0)
    sys.exit(os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode")
             else (status >> 8))


if __name__ == "__main__":
    main(sys.argv[1:])
