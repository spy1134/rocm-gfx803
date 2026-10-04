# GPU hang capture

Host-side tools for catching a gfx803 (Polaris) card in the act when it hangs
under sustained compute load. They run on the host, outside any container, and
only read: nothing here halts waves or writes to the GPU.

## Why this exists

A hung gfx803 compute queue usually takes the display and often the whole host
down with it within seconds, and journald loses the last minutes before a hard
reset. What is left after the reboot says that something hung, not what. These
tools record the card's state continuously and, at the moment a queue stops
making progress, save the queue registers and the dispatch packets that were in
flight, so the stuck kernel can be identified afterwards. That is how the binary
Winograd convolution kernels behind
[`rocm6.4.4/patches/miopen/winograd-bin-gfx803-disable.patch`](../../rocm6.4.4/patches/miopen/winograd-bin-gfx803-disable.patch)
were found.

## The tools

- `gpu_monitor.sh`: logs temperature, power, fan, shader and memory clock, busy
  percentage and VRAM use every 2 seconds, syncing after every line so the log
  survives a hard hang up to its last sample.

      bash gpu_monitor.sh [logfile]        # default ~/gpu_monitor.log

- `gpu_hang_watch.sh`: run as root in its own terminal. It captures evidence when
  the kernel logs a GPU timeout, failed queue preemption or reset, or when a
  busy compute queue's read and end-of-pipe pointers stay frozen for
  `PROGRESS_SECONDS` (default 4). Kernels in these workloads finish in
  milliseconds, so a frozen queue is a hang whatever power or clocks show.

      sudo bash gpu_hang_watch.sh [output_dir]   # default /var/log/gpu_hang

  Each capture lands in `<output_dir>/<timestamp>/`: the trigger, kernel
  messages, the KFD queue registers and runlist, the decoded stuck queue with the
  in-flight kernels' code, fence state, the processes holding `/dev/kfd`, and a
  devcoredump when the driver made one.

- `gpu_hang_decode.py`: used by the watcher at capture time. It parses the gfx8
  `CP_HQD` registers from `/sys/kernel/debug/kfd/hqds` and reads the AQL ring
  from the owning process's memory, recording each dispatch's launch shape and
  arguments and saving the kernel's code. By hand:

      sudo python3 gpu_hang_decode.py --selftest --outdir DIR   # during a run
      python3 gpu_hang_decode.py --hqds FILE                    # a saved capture
      sudo python3 gpu_hang_decode.py --state                   # busy queues' pointers

  Disassemble saved code with
  `llvm-mc --disassemble -triple=amdgcn -mcpu=gfx803`.

Both scripts use the first AMD GPU they find; set `PCI=<domain:bus:device.function>`
to pick another.

## Cautions

- Never read `/sys/kernel/debug/kfd/mqds` while queues are active on this kernel:
  it oopses (a NULL call in `pqm_debugfs_mqds`) and turns a GPU hang into a host
  lockup. `hqds` and `rls` are safe, and are all these tools read.
- `UMR_WAVES=1` adds umr wave dumps to each capture. Halting waves can itself
  hang a healthy GPU, so it is off by default.
- With a desktop on the same card, amdgpu resets the GPU about 10 seconds into a
  hang, and the stuck queue is gone after the reset. Keep the trigger windows well
  under that.
- A host that freezes outright (for example a data fabric sync flood) can stop
  within one sample, before any capture. The next boot's
  `journalctl -b -1 -k` and `journalctl -b 0 -k | grep -E 'reset reason|Hardware Error'`
  are the evidence for those.
