#!/bin/bash
# opinfo_compare_cpu.sh [--fp16] [WORKDIR] [IMAGE]: PyTorch's own OpInfo check of every operator on the GPU against the CPU
# (test_ops.py TestCommonCUDA.test_compare_cpu_*: each op's reference inputs, one dtype per op, atol and rtol 1e-3).
# The test file and its Python dependencies (pytest, expecttest, hypothesis) are fetched into WORKDIR (default
# ./opinfo-work), for the PyTorch commit the image was built from; the image itself is not changed.
#
# Tests run 25 to a process with 60% of the VRAM available to them. When a process dies (a device trap, a hang that
# hits the 25-minute limit), its tests run again one per process, so a crash names its op and costs only that op.
# Results: WORKDIR/out/results.txt, one "RESULT test status seconds detail" line per test; WORKDIR/out/log.txt.
#
# --fp16 runs opinfo_fp16.py instead: every op that has float16 on the GPU, over its reference inputs, against float32
# on the CPU (atol and rtol 1e-2), all in one process that resumes after the op that killed it. Results in
# WORKDIR/out/fp16.txt. Expected failures there: ops with random output (random sampling, dropout, rrelu, fractional
# max pooling, empty tensors), integer casts of inf, NaN or out-of-range values, sort/argsort/mode index order among
# equal values, float16 rounding or overflow where float32 does not (nan_to_num's posinf default, matrix_exp,
# mvlgamma, round with decimals), and ops with a different default per backend (batch norm's saved statistics).
set -u
FP16=0; [ "${1:-}" = "--fp16" ] && { FP16=1; shift; }
S=$(cd "$(dirname "$0")" && pwd)
W=$(realpath -m "${1:-opinfo-work}"); IMAGE=${2:-rocm-gfx803:rocm6.4.4}
mkdir -p "$W/deps" "$W/src" "$W/out"
run() { timeout "${TO:-1500}" docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/w/deps -e PYTORCH_TEST_WITH_SLOW=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$S:/suite:ro" -v "$W:/w" --entrypoint python3 "$IMAGE" -u "$@"; }

if [ ! -s "$W/src/test_ops.py" ]; then
  commit=$(docker run --rm --entrypoint python3 "$IMAGE" -c "import torch; print(torch.version.git_version)")
  curl -fsSL -o "$W/src/test_ops.py" "https://raw.githubusercontent.com/ROCm/pytorch/$commit/test/test_ops.py" \
    || { echo "cannot fetch test_ops.py for PyTorch commit $commit" >&2; exit 1; }
fi
if [ ! -d "$W/deps/expecttest" ]; then
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$W:/w" --entrypoint pip "$IMAGE" \
    install -q --target /w/deps pytest expecttest hypothesis || exit 1
fi
if [ "$FP16" = 1 ]; then
  run -c "
from torch.testing._internal.common_methods_invocations import op_db
print('\n'.join(o.formatted_name for o in op_db))" > "$W/out/fp16-ops.txt"
  : > "$W/out/fp16.txt"
  while :; do
    mapfile -t L < <(grep -vxFf <(awk '{print $2}' "$W/out/fp16.txt") "$W/out/fp16-ops.txt")
    [ ${#L[@]} -eq 0 ] && break
    run /suite/opinfo_fp16.py "${L[@]}" > "$W/out/f.tmp" 2> "$W/out/f.err"
    grep "^RESULT" "$W/out/f.tmp" >> "$W/out/fp16.txt"
    grep -q GROUPDONE "$W/out/f.tmp" && break
    # the process died on the first remaining op it had not reported
    for op in "${L[@]}"; do
      grep -q "^RESULT $op " "$W/out/f.tmp" || { echo "RESULT $op DIED 0s $(tail -c 200 "$W/out/f.err" | tr '\n' ' ')" >> "$W/out/fp16.txt"; break; }
    done
  done
  awk '{print $3}' "$W/out/fp16.txt" | sort | uniq -c
  grep -E ' (FAIL|ERROR|DIED) ' "$W/out/fp16.txt" | cut -c1-200
  exit 0
fi
[ -s "$W/out/tests.txt" ] || run -c "
import sys; sys.path.insert(0, '/w/src'); import test_ops
print('\n'.join(n for n in sorted(dir(test_ops.TestCommonCUDA)) if n.startswith('test_compare_cpu_')))" > "$W/out/tests.txt"

: > "$W/out/results.txt"
mapfile -t T < "$W/out/tests.txt"
echo "$(date +%H:%M) ${#T[@]} tests" > "$W/out/log.txt"
for ((i = 0; i < ${#T[@]}; i += 25)); do
  G=("${T[@]:i:25}")
  run /suite/opinfo_group.py "${G[@]}" > "$W/out/g.tmp" 2> "$W/out/g.err"
  grep "^RESULT" "$W/out/g.tmp" >> "$W/out/results.txt"
  if ! grep -q GROUPDONE "$W/out/g.tmp"; then
    echo "tests $i-$((i + 24)): the process died, running them one by one" >> "$W/out/log.txt"
    for t in "${G[@]}"; do
      grep -q "^RESULT $t " "$W/out/results.txt" && continue
      TO=600 run /suite/opinfo_group.py "$t" > "$W/out/s.tmp" 2> "$W/out/s.err"
      if grep -q "^RESULT" "$W/out/s.tmp"; then grep "^RESULT" "$W/out/s.tmp" >> "$W/out/results.txt"
      else echo "RESULT $t DIED 0s $(tail -c 300 "$W/out/s.err" | tr '\n' ' ')" >> "$W/out/results.txt"; fi
    done
  fi
done
echo "$(date +%H:%M) done" >> "$W/out/log.txt"
awk '{print $3}' "$W/out/results.txt" | sort | uniq -c
grep -E ' (FAIL|ERROR|DIED) ' "$W/out/results.txt" | cut -c1-240
