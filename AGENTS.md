# Agent instructions for rocm-gfx803

Read `README.md` first for the layout and the results, and `docs/STACK.md` for
how the images are built and run. This file gives the process rules and the
judgment calls. You cannot get either one
from the code or from a patch header alone.

## Scope

This repository maintains one stack: ROCm 6.4.4 for gfx803, with its images
(`Dockerfile`, `Dockerfile-ComfyUI`, `Dockerfile-Ollama`), its patches
(`patches/`), and the host tools in `tools/`. It builds with plain
`docker build` from the repository root, has no CI, and is validated on a
local card. `docs/FINDINGS.md` holds the findings, `docs/OPTIMIZATION_TARGETS.md`
the performance work, and `docs/KERNEL_BUGS.md` the earlier investigations from
Schaka/rocm-gfx803, which this project started from. Work on another ROCm
version goes on its own branch.

## Standing philosophy

### Build locally. Never push.

All work in this repo builds locally. You build on the host or on the box at
192.168.1.214. Do not trigger remote CI. Do not push. Do not commit. Do these
only when the user asks for that one commit or push. Committing and pushing are
the user's jobs. A remote workflow builds whatever the remote branch holds at
that moment, so a push is a prerequisite for it anyway. The rule is
shorter than that: do not. Build locally, test locally, and give the user the
diff or the local build. Then they decide. This rule also covers
`gh workflow run`. Starting a remote run without being asked is the same
violation as a push.

On the one commit the user does ask for, write it as a human alone wrote it.
Never add a `Co-Authored-By: Claude ...` trailer or any other AI-attribution
line, in this repo or any other.

### Fix at the source. Do not use workarounds.

Fix a gfx803 bug where the bug lives. That can be the broken Tensile logic, the
broken MIOpen solver, or the broken MIGraphX pass. Do not hide it with a retry
loop. Do not hide it with a narrower input-shape rule that avoids the trigger.
Do not hide it with a fallback to a slower path that you picked because you were
afraid, without proof that the fast path is broken. Do not hide it with a
try/catch that swallows the failure. Some bugs have no source fix. The code is
upstream and you do not control it, or the hardware cannot do it. Say that in
plain words and stop. Do not ship a workaround that looks like a fix.
`wgm-miscompute.sh` is the model. It found the exact broken instruction
sequence, and it fixed the one parameter that Tensile computes wrong. It did not
avoid the shapes that show the bug.

### Correctness costs speed sometimes. That is fine, to a point.

A correctness fix that costs 5 to 10 percent of throughput is an acceptable and
expected trade. Do not negotiate around it. Beyond that, say so and ask. Do not
accept a large regression in silence, and do not pick a workaround in silence to
avoid one. Never trade correctness for speed without saying it out loud first.

### Comments say WHY. They never say WHAT.

Write code that is clear enough to read on its own. A comment that repeats what
the next three lines do is dead weight, and it becomes wrong the first time the
code changes. Write a comment only for a reason you cannot see in the code: why
this bound, why this order, why a thing that looks like a workaround is actually
correct here, and what broke the last time someone tried the obvious approach.
The patch headers in this repo are the model. Each one states WHY (how the bug
was found and what it looks like) before WHAT (the diff itself).

### A comment is not commit history.

Do not write "changed X to Y", "added this for the Z fix", or "removed unused
W". `git log` and `git blame` do that job, and such a comment becomes wrong as
soon as a later edit lands. A comment must make sense to a reader who has no
idea what the last edit was.

### Never write a comment about something that is absent.

Do not explain that something *is not* there, *used to be* there, or *is not
done*. Examples of bad comments: "no libomp-dev here", "removed the X
workaround", "we do not do Y anymore". A reader sees only the code that exists.
A reader cannot test a note about code that does not exist, so it is dead weight
forever. A dropped line needs no comment, because its absence is visible. Write
a comment only to justify something that is present.

### A comment does not describe the whole system.

Some comments need to explain how one file relates to three others, why a
Dockerfile stage exists in the larger pipeline, or how a patch fits the
gfx803-versus-mainline story. That text belongs in `docs/STACK.md`, `README.md`
or a patch header. It does not belong in an inline code comment. A code comment explains
the code next to it. If you want to write three paragraphs above a function
about how the pipeline works, that is a docs change, not a comment.

### Docs and patch headers describe the bug and the fix, never the debugging session.

A patch header, a `README.md` section, or any other doc does one job. It
tells a reader with no memory of the work why the code exists and what the
fix does. State plainly what breaks without the patch. For example, without
this patch the driver never frees VRAM until the process exits. When
reproducing the bug helps verify the fix, include the steps to reproduce it.

A doc never logs the investigation. Do not write "initially this looked
like X" or "CORRECTION: testing this against Y showed Z" or a dated "later
that day we found...". Do not frame the explanation around one person's bug
report, such as "a ComfyUI user reported" or "issue #2's second comment
attached this repro". A report only triggered the work, not what the fix
targets. Describe the bug and the fix on their own terms, as if you found
the bug yourself with no report at all. A report's URL can appear as a
citation, never as the spine of the explanation.

When a doc turns out wrong or incomplete, rewrite it in place instead of
appending a correction. Replace the wrong text so the doc reads as one
coherent, current explanation. Never leave the old account standing next to
the new one for a reader to reconcile. `git log` and `git blame` already
hold every past version. That history is what makes overwriting the doc's
own text safe.

### A doc aimed at a person never points at an agent-only file.

`README.md`, a `BUILD.md`, a `NOTES.md`, a patch header, a tool's docstring, and
any message a tool prints are all written for a person. Never cite `AGENTS.md`,
`CLAUDE.md`, a skill file, or any other file whose only reader is an agent: a
person following this repo has no reason to open one, and sending them there
hands them instructions that were never meant for them. If a rule that lives in
an agent file is what makes a human-facing doc correct, restate the rule in that
doc, in its own words. The same goes for a `::error::` line or a warning a script
prints at build time: state the reason itself, never a pointer at a file the
reader may not have.

## Why this repo exists, and what that means for your work

AMD dropped support for gfx803 (Polaris) in ROCm 6.0. Every fix here is local.
Nothing goes upstream to AMD, and a ROCm version bump fixes nothing unless you
go and look. Two results follow.

1. Assume nothing is fixed until you read the pinned source. A patch header
   that says "confirmed broken as of ROCm X" is one moment in time, not a
   permanent fact. Before you re-diff a patch or investigate again, grep the
   pinned commit for the target function or struct and read what is there now.
   This repo's history has both outcomes. Some patches were already obsolete,
   because the fix landed upstream on its own. In others the target code was
   replaced by something new whose behavior on the same bug class is unknown
   and unverified. That case is not fixed and not proven broken. It is unknown
   until you read the pinned source.
2. A patch that applies cleanly proves nothing about correctness. The common
   bug class on this hardware is a silent wrong answer. `rocblas_status_success`
   returns with bad numbers, or a kernel dispatches fine and returns garbage.
   Re-diffing a patch so it compiles again is necessary and not enough. Say so
   in the patch or the notes: "NOT YET RE-VERIFIED ON REAL HARDWARE". Keep that
   note until someone runs the original repro against the new binaries, instead
   of only confirming that the diff applies.

## Investigation workflow

These steps worked many times in this repo's history.

1. Trace before you patch. `MIOPEN_ENABLE_LOGGING_CMD=1 MIOPEN_LOG_LEVEL=6` and
   `MIGRAPHX_TRACE_COMPILE=1` show the solver and op graph that really ran. They
   do not show what you guess from the Dockerfile or the op name. Several
   "obvious" explanations here were wrong after a trace. One ConvTranspose
   investigation blamed a specific MIOpen solver. The trace showed a different
   code path: MIGraphX's own graph rewrite, and MIOpen was not involved.
2. Isolate before you conclude. When a bug shows through the whole stack (ORT to
   MIGraphX to MIOpen to rocBLAS), test each layer alone before you pick a
   suspect. `MIOpenDriver <op> ... -V 1` runs MIOpen's own GPU-versus-CPU
   reference comparison with no ORT or MIGraphX in the process. It cleared MIOpen
   in favor of a layer above it, and the other way around, more than once.
3. Compare against another architecture before you call a bug gfx803-specific.
   A failure that also happens on a newer arch (gfx900+ or gfx12x) with the same
   source pin is an upstream bug, and it is not gfx803's to fix here. Report it
   upstream. A failure only on gfx803 with identical source is a fair target for
   a local patch. Make this distinction before you write the patch, not after. A
   gfx803-only patch for a generic bug helps nobody else who hits the same code
   path, and that work is not this repo's job.
4. Turn patches off one at a time before you credit one. When several patches
   can explain a symptom, build one variant per disabled patch and keep
   everything else the same. History: two rocBLAS patches were each *partly*
   responsible. Removing them made the symptom much worse, which is a different
   and more useful result than "no difference" or "yes".
5. When one version works and another does not, first ask whether the two are
   comparable. Newer ORT, ROCm, and MIGraphX add real new features (new ONNX
   opsets, new parser paths) that the older version cannot run at all. A test
   that fails on the new version and does not even load on the old one is not a
   regression. It is new and still broken. Count something as "worked there,
   broken here" only after you make sure that the older version ran the same
   code path and got it right.

## Patch conventions

- Each patch header states WHY before WHAT: how the bug was found, what it looks
  like, and the hardware measurements where they apply. If you cannot write a
  convincing WHY, your investigation is not finished.
- Two apply styles exist on purpose. Use `git apply` for a tree that is a real
  git repo, such as `rocm-systems`. Use `patch -p1` for a sparse checkout of a
  monorepo subdirectory, because `git apply --check` there reports success and
  changes nothing ("Skipped patch", exit 0) on this box's git version. Match the
  style that the other patches in that directory already use.
- Every `.sh` driver makes sure that its own apply worked. It greps for a marker
  string, and it stops with `exit 1` when the marker is absent. Do not add a driver
  that trusts only the exit code of the patch tool.
- Each ROCm version lives on its own branch with its own patch set. Port a fix
  to another branch by copying the patch and re-verifying it on the card there;
  a patch that applies on both is not proof that it fixes both, and a shared
  file risks a half-fixed bug on one branch reaching the hardware-tested state
  of another.

## Hardware access

Real-hardware validation needs a real gfx803 card. You cannot emulate it, and
another GPU will not do. If you do not have one, say so. Do not report a patch as
"verified on hardware" because it only built and applied cleanly. The container needs
`--device=/dev/kfd --device=/dev/dri --group-add video` passed through. Inside
the container, `rocminfo` must list the card as a `KERNEL_DISPATCH` agent before
you trust anything else it reports.

Known traps when you add instrumentation on real hardware:

- Device-side `printf` can hang the kernel instead of only slowing it down.
- Any extra device-side write for debugging changes timing, and it can hide the
  race you are looking for.
- Always make sure that you run against the patched binary you think you are
  testing. A stale image or container is a repeated false lead here. Make sure
  of that, and not only that the patch file on disk looks right.

## What does not need the card

You can answer these without hardware. Does the Dockerfile build. Does a patch
apply against a given pin. Where does a bug live, read from the actual pinned
source, diffed across ROCm versions, or followed through the code that really
calls a compiler pass instead of the code you assume calls it. You can also set
up a cross-arch comparison test without gfx803, because that step needs a card of
some kind and not necessarily this one. Do that first, to learn whether a bug is
generic, before you spend hardware time on the gfx803 side of the comparison.
