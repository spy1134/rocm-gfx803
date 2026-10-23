// Which load paths read stale data across a dispatch boundary on gfx803, and which
// kernel-entry cache instruction, if any, cures each one.
//
// Every trial allocates a 4 KiB buffer, fills it with a new value by a host copy
// (a shader copy through L2 at this size), lets 144 blocks sum it, and frees it, so
// the next trial gets the same page back through ROCr's fragment heap. The output
// is kept small on purpose: a large read-back evicts the L1 and hides the bug.
//
// Build: hipcc --offload-arch=gfx803 -O2 loadpaths.hip -o loadpaths
// Run:   CLR_GFX8_TC_INVALIDATE=0 ./loadpaths   (the runtime's per-dispatch invalidate off)
//        ./loadpaths                            (on; every cell must be 0)
// Exits 1 when any trial read stale data.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP %s at line %d\n", hipGetErrorString(e), __LINE__); exit(2); } } while (0)

typedef int int32x4_t __attribute__((ext_vector_type(4)));
__device__ float raw_buffer_load_f32(int32x4_t rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.f32");

enum Kind { FLAT, BUFFER, BUFFER_NV, S_LOAD, S_BUFFER_LOAD, S_BUFFER_LOAD_NV, NKIND };
enum Entry { NONE, WBINVL1_VOL, WBINVL1, DCACHE_INV, NENTRY };
static const char* kind_name[NKIND] = {"flat_load", "buffer_load, NV clear", "buffer_load, NV set", "s_load", "s_buffer_load, NV clear", "s_buffer_load, NV set"};
static const char* entry_name[NENTRY] = {"none", "wbinvl1_vol", "wbinvl1", "dcache_inv"};

// Descriptor word3: float, data format 32. Bit 27 is the low bit of the memory-type field in
// ROCr's gfx8 register layout (NV in AMD's GCN3 ISA guide); Tensile and MIOpen leave it clear.
constexpr int WORD3 = 0x00027000, WORD3_NV = 0x08027000;

__global__ void sum(const float* in, float* out, int n, int kind, int entry) {
    // buffer_wbinvl1_vol is the invalidate LLVM's gfx8 memory model emits for an acquire.
    if (entry == WBINVL1_VOL) asm volatile("buffer_wbinvl1_vol" ::: "memory");
    if (entry == WBINVL1) asm volatile("buffer_wbinvl1" ::: "memory");
    if (entry == DCACHE_INV) asm volatile("s_dcache_inv\n\ts_waitcnt lgkmcnt(0)" ::: "memory");
    unsigned long long p = (unsigned long long)in;
    int32x4_t rsrc = {__builtin_amdgcn_readfirstlane((int)(unsigned)p), __builtin_amdgcn_readfirstlane((int)(unsigned)(p >> 32)),
                      __builtin_amdgcn_readfirstlane(n * 4), __builtin_amdgcn_readfirstlane(kind == BUFFER_NV || kind == S_BUFFER_LOAD_NV ? WORD3_NV : WORD3)};
    float s = 0.f;
    if (kind >= S_LOAD) {
        // One wave reads every element through the scalar cache; only lane 0 reports.
        for (int i = 0; i < n; ++i) {
            float v;
            if (kind == S_LOAD)
                asm volatile("s_load_dword %0, %1, 0x0\n\ts_waitcnt lgkmcnt(0)" : "=s"(v) : "s"(p + i * 4ull));
            else
                asm volatile("s_buffer_load_dword %0, %1, %2\n\ts_waitcnt lgkmcnt(0)" : "=s"(v) : "s"(rsrc), "s"(i * 4));
            s += v;
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) s += kind == FLAT ? in[i] : raw_buffer_load_f32(rsrc, i * 4, 0, 0);
        for (int o = 32; o > 0; o >>= 1) s += __shfl_down(s, o);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// A line that survives the boundary is hit by any later load of its address, whatever
// path that load takes, so every cell starts from caches emptied on every CU.
__global__ void scrub() { asm volatile("buffer_wbinvl1\n\ts_dcache_inv\n\ts_waitcnt lgkmcnt(0)" ::: "memory"); }

static int stale_trials(int kind, int entry, int trials, int n) {
    const int blocks = 144;
    std::vector<float> h(n), r(blocks);
    int bad = 0;
    scrub<<<4096, 64>>>();
    for (int t = 0; t < trials; ++t) {
        float *in, *out;
        CK(hipMalloc(&in, n * 4));
        CK(hipMalloc(&out, blocks * 4));
        float v = (float)(t % 97 + 1);
        for (int i = 0; i < n; ++i) h[i] = v;
        CK(hipMemcpy(in, h.data(), n * 4, hipMemcpyHostToDevice));
        sum<<<blocks, 64>>>(in, out, n, kind, entry);
        CK(hipMemcpy(r.data(), out, blocks * 4, hipMemcpyDeviceToHost));
        for (int b = 0; b < blocks; ++b)
            if (r[b] != v * n) { ++bad; break; }
        CK(hipFree(in));
        CK(hipFree(out));
    }
    return bad;
}

int main(int argc, char** argv) {
    int trials = argc > 1 ? atoi(argv[1]) : 1000, n = 1024;
    const char* knob = getenv("CLR_GFX8_TC_INVALIDATE");
    printf("stale trials of %d, 4 KiB recycled buffer, CLR_GFX8_TC_INVALIDATE=%s\n", trials, knob ? knob : "(unset)");
    printf("%-25s", "load \\ kernel entry");
    for (int e = 0; e < NENTRY; ++e) printf("%12s", entry_name[e]);
    printf("\n");
    int total = 0;
    for (int k = 0; k < NKIND; ++k) {
        printf("%-25s", kind_name[k]);
        for (int e = 0; e < NENTRY; ++e) {
            int bad = stale_trials(k, e, trials, n);
            total += bad;
            printf("%12d", bad);
            fflush(stdout);
        }
        printf("\n");
    }
    return total ? 1 : 0;
}
