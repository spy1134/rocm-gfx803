# Console mouse

`console_mouse.py` lets a terminal UI use the mouse on a Linux text console
(Ctrl+Alt+F2 and up, no desktop running), also inside screen or tmux and
through `docker exec -it`.

## Why this exists

Terminal UIs such as Textual read the mouse as xterm escape sequences, which a
desktop terminal emulator writes into their input. The Linux console never
writes them. gpm reads the mouse on the console, but it only serves programs
linked against libgpm that run directly on the console, and its library
refuses a program behind a pseudo-terminal. So a UI in a container, or in
screen, gets no mouse on a text console even with gpm running.

## What it does

    console_mouse.py [--log FILE] [--] COMMAND [ARGS...]

It runs COMMAND on its own pseudo-terminal and passes the keyboard and the
screen through unchanged. While COMMAND has mouse reporting switched on, the
wrapper connects to gpm's socket (`/dev/gpmctl`) for the active console and
writes clicks, drags, releases and wheel steps into COMMAND's input as xterm
reports, in the encoding COMMAND asked for (SGR or legacy X10). While
reporting is off it stays disconnected, and gpm's own copy and paste works as
usual. gpm still draws the pointer.

On a desktop, without gpm, or with no terminal on stdin, it just runs COMMAND.

`rocm6.4.4/gfx803` runs its UI through it.

## Requirements and limits

- gpm running (`systemctl enable --now gpm`) and Python 3 on the host.
- You must own the console you are on, which you do after logging in there:
  gpm accepts a client for a console only from that console's owner.
- Positions are console cells. They line up only while COMMAND fills the
  whole console: one full-size screen window, no status or caption line.
- A program that never switches mouse reporting on gets nothing.

## Checking a mouse

`--log FILE` appends every gpm event and the bytes sent for it. Use it to check
the wheel direction and the button mapping of a new mouse.
