#!/bin/sh
# Compile an OpenCL kernel through libamd_comgr, as hipRTC and MIOpen do, and make sure that a non-inlined function
# which casts a __local address to a generic pointer reads the shared aperture base through the implicit-argument
# pointer in s[8:9]. libamd_comgr embeds its own compiler, so a clang-19 that is patched says nothing about it.
#
# Needs no GPU. Usage: check-comgr.sh [directory holding the libamd_comgr.so.3 to test]
set -eu

LIBDIR="${1:-/opt/rocm/lib}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/k.cl" <<'CL'
generic int *g;
__attribute__((noinline)) void callee(__local int *p, int i) { g = (generic int *)(p + i); }
__kernel void k(__global int *o) {
  __local int a[64];
  int t = __builtin_amdgcn_workitem_id_x();
  callee(a, t);
  *g = t;
  __builtin_amdgcn_s_barrier();
  o[t] = a[63 - t];
}
CL

cat > "$WORK/t.c" <<'C'
#include <amd_comgr/amd_comgr.h>
#include <stdio.h>
#include <stdlib.h>
#define CK(s, m) do { if ((s) != AMD_COMGR_STATUS_SUCCESS) { fprintf(stderr, "comgr failed: %s\n", m); exit(1); } } while (0)
int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
  char *src = malloc(n); if (fread(src, 1, n, f) != (size_t)n) return 1; fclose(f);
  amd_comgr_data_t d; amd_comgr_data_set_t in, bc, rel, out; amd_comgr_action_info_t a;
  CK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &d), "create_data");
  CK(amd_comgr_set_data(d, n, src), "set_data");
  CK(amd_comgr_set_data_name(d, "k.cl"), "set_data_name");
  CK(amd_comgr_create_data_set(&in), "create_data_set");
  CK(amd_comgr_data_set_add(in, d), "data_set_add");
  CK(amd_comgr_create_action_info(&a), "create_action_info");
  CK(amd_comgr_action_info_set_language(a, AMD_COMGR_LANGUAGE_OPENCL_2_0), "set_language");
  CK(amd_comgr_action_info_set_isa_name(a, "amdgcn-amd-amdhsa--gfx803"), "set_isa_name");
  const char *opts[] = {"-O3", "-mcode-object-version=5"};
  CK(amd_comgr_action_info_set_option_list(a, opts, 2), "set_option_list");
  CK(amd_comgr_create_data_set(&bc), "create_data_set");
  CK(amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, a, in, bc), "source to bc");
  CK(amd_comgr_create_data_set(&rel), "create_data_set");
  CK(amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, a, bc, rel), "bc to relocatable");
  CK(amd_comgr_create_data_set(&out), "create_data_set");
  CK(amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, a, rel, out), "relocatable to executable");
  amd_comgr_data_t e; size_t sz = 0;
  CK(amd_comgr_action_data_get_data(out, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &e), "get_data");
  CK(amd_comgr_get_data(e, &sz, NULL), "get size");
  char *buf = malloc(sz); CK(amd_comgr_get_data(e, &sz, buf), "get bytes");
  FILE *o = fopen(argv[2], "wb"); fwrite(buf, 1, sz, o); fclose(o);
  return 0;
}
C

gcc -I/opt/rocm/include "$WORK/t.c" -o "$WORK/t" -L"$LIBDIR" -lamd_comgr -Wl,-rpath,"$LIBDIR"
"$WORK/t" "$WORK/k.cl" "$WORK/k.co"
/opt/rocm/lib/llvm/bin/llvm-objdump -d --mcpu=gfx803 "$WORK/k.co" > "$WORK/k.s"
sed -n '/<callee>:/,/s_setpc/p' "$WORK/k.s" > "$WORK/callee.s"
[ -s "$WORK/callee.s" ] || { echo "FATAL: no callee in the code object; the probe was inlined" >&2; exit 1; }
if grep -q 's_mov_b64 s\[[0-9:]*\], 0xc4' "$WORK/callee.s"; then
    echo "FAIL: libamd_comgr in $LIBDIR loads the shared aperture from address 0xc4 in a callee" >&2
    exit 1
fi
grep -q 's_load_dword s[0-9]*, s\[8:9\], 0xc4' "$WORK/callee.s" \
    || { echo "FATAL: the callee does not read the shared aperture through s[8:9]" >&2; exit 1; }
echo "OK: libamd_comgr in $LIBDIR reads the implicit arguments of a callee through s[8:9]"
