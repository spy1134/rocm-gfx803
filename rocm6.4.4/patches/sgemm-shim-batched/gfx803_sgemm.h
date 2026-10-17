// Replacement GEMM kernels for gfx803, used by sgemm_shim.cpp in place of
// rocBLAS/Tensile's own kernels: fp32, and the same kernels with fp16 storage
// and fp32 arithmetic for the fp16 calls the shim takes over (see its header).
//
// WHY
// ---
// rocBLAS's Tensile fp32 GEMMs return wrong results on gfx803 with
// rocblas_status_success. The first call on a fresh handle is right; later
// calls are intermittently wrong, in anywhere from ~1% to ~100% of output
// elements, for every shape class tried (8x8x8, 40x72x800, 128x128x1024,
// 512^3, 768x768x3072) and for all 55 solutions rocblas_gemm_ex_get_solutions
// returns. rocBLAS built from rocm-5.7.0, the last release that supported
// gfx803, does the same. The cause is rocBLAS's device memory pool: it reuses
// memory as GlobalSplitU (split-K) scratch without re-zeroing it, and gfx803's
// split-K reduction accumulates into that scratch with a software
// compare-and-swap loop (GCN3 has no float atomic add) that assumes it starts
// at zero. Running Tensile's kernels on a private handle with a self-zeroed
// workspace is correct too, but measured slower than this kernel at every
// size up to 2048^3, because of the per-call zeroing.
//
// Without this kernel, SD1.5 LoRA training on gfx803 trains on wrong numbers:
// attention backward (torch.bmm/baddbmm, rocblas_*gemm_strided_batched*) at
// 8x64x64, d=160 is wrong on 4 or 5 of 5 runs, and training loss sits near
// 3-4 instead of 0.1-0.2. To reproduce, compare that attention's forward and
// backward on the GPU against a float64 CPU reference over several runs with
// fresh inputs, with GFX803_SGEMM_SHIM_DISABLE=1 and without.
//
// WHAT
// ----
// Two tile routines, both 16x16-thread workgroups with A and B staged through
// LDS one 16-deep K slice at a time, and no split-K and no scratch memory, so
// the failure above has nothing to act on:
//   gfx803_sgemm_tile:      a 16x16 tile of C, one output per thread.
//   gfx803_sgemm_tile_p<P>: a (16P)x(16P) tile of C, a PxP block of outputs
//                           per thread; P = 2 and P = 4 are used.
// gfx803_sgemm_tile_p_fast and gfx803_sgemm_tile_p2_fast are the same 64x64
// and 32x32 tiles with B not transposed, which is most of what PyTorch
// issues; see the comment at gfx803_sgemm_tile_p_fast.
// gfx803_sgemm_tile_g is the same tile again for the transposed layouts that
// rocBLAS sees from nn.Linear and its weight gradients; see the comment there.
// All of them take fp32 or fp16 operands (E below): fp16 values are widened to
// fp32 when they are stored to LDS (gfx803_stage_d says why not earlier), so
// LDS, the multiply-adds and the sums are the fp32 kernel's, and C is rounded
// to fp16 once. Polaris has no packed
// fp16 arithmetic, so fp16 cannot be faster per multiply-add than fp32; what
// the fp16 path gains is the kernel itself. rocBLAS's fp16 kernels for gfx803
// keep fp16 in LDS and widen inside the inner loop (10 v_cvt_f32_f16 per 16
// v_mac_f32 in its 128x32 tile) and load operands as 2-byte flat_load_ushort,
// which is why they run SD1.5/SDXL's linears 1.3-1.9x slower than the fp32
// kernels here (2560x8192x320: 3.0 against 4.1 TFLOPS). Widening 16 values per
// 512 multiply-adds at the LDS store costs about 3%.
// gfx803_w_kernel computes a 64x64 (or 64x32) tile with one wavefront, an 8x8
// (or 8x4) block per thread, and no barrier; it takes the fp16 batched calls
// that gfx803_w_applies picks from the 64x64 tiles.
// gfx803_w2_kernel computes a 64x128 tile with two wavefronts sharing the A
// piece; it takes the fp16 TN calls (nn.Linear) that gfx803_w2_applies picks.
// gfx803_sgemm_small_n_kernel takes GEMMs with at most 8 columns of C and
// op(A) = T, a token-by-token decode step's nn.Linear, as a GEMV; it sums in
// a different order than the tiles (see the comment there).
// Every output accumulates the same products in the same order in all the
// tiles, so they return bit-identical results and gfx803_sgemm_pick_tile picks
// a tile size on speed alone. Each serves two entry points: a single GEMM,
// and a strided-batched one with the batch index in blockIdx.z and each
// matrix offset by its stride.
//
// The one-output-per-thread tile alone reaches 1.1-1.2 TFLOPS on an RX 590,
// about stock rocBLAS's speed, and spent 88% of SD1.5 LoRA training's GPU time
// at that rate. The larger tiles read each LDS value into P*P multiply-adds
// instead of one; the 64x64 tile reaches 2.5-3 TFLOPS on the large
// conv-as-GEMM and linear shapes (512x4608x8192: 33.7 -> 13.1 ms), and the
// 32x32 tile serves the shapes with too few 64x64 tiles to fill the card
// (1280-channel convs at the smallest latent size, LoRA's rank-8 GEMMs).
// Training steps drop from 7.8 to about 4.3 s. The fast variants take the
// large shapes to 3.8-4.8 TFLOPS (2048^3: 3.3 -> 4.8; 320x4096x2880:
// 3.0 -> 4.35).
//
// Memory access is laid out for the column-major operands rocBLAS passes: see
// the comment in gfx803_sgemm_tile. Arranged the other way, a leading
// dimension of 4096 or 8192 floats put whole wavefronts and every concurrent
// workgroup on one DRAM channel: attention scores at 4096 tokens took 214 ms
// (11 ms at 4100 tokens), and MIOpen's GEMM-based 3x3 convolutions ran 6-9x
// slower.
//
// Launchers split the work into launches the hardware can take (C's columns
// over the grid Y limit, the batch over grid Z) and return hipError_t, so a
// launch that still fails reaches the shim, which falls back to rocBLAS.
#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#define GFX803_SGEMM_TILE 16
#define GFX803_SGEMM_MAX_GRID_YZ 65535

// C[MxN] = alpha * op(A) * op(B) + beta * C, column-major storage (BLAS/
// rocBLAS convention), matching rocblas_sgemm/rocblas_gemm_ex semantics:
//   transA/transB: false = N (no transpose), true = T (transpose)
//   A is (transA ? K x M : M x K) with leading dimension lda, column-major
//   B is (transB ? N x K : K x N) with leading dimension ldb, column-major
//   C is M x N with leading dimension ldc, column-major
template <typename T>
__device__ __forceinline__ void gfx803_sgemm_tile(int M, int N, int K,
                                                  float alpha,
                                                  const T* __restrict__ A, int lda, bool transA,
                                                  const T* __restrict__ B, int ldb, bool transB,
                                                  float beta,
                                                  T* __restrict__ C, int ldc) {
    // threadIdx.x always walks the dimension that is contiguous in memory:
    // C's rows (column-major), and for each operand whichever of its two
    // indices is contiguous given its transpose flag. With the lanes of a
    // wavefront striding by the leading dimension instead, a leading
    // dimension of 4096 or 8192 floats puts all 16 lanes in one DRAM channel
    // and serializes them: attention scores at 4096 tokens ran 18x slower
    // than at 4100. The +1 columns keep the transposed LDS accesses (As[tx][k]
    // reads, Bs[tx][ty] writes) free of bank conflicts. Each output still
    // accumulates the same products in the same order, so results are
    // bit-identical to the row-walking layout.
    __shared__ float As[GFX803_SGEMM_TILE][GFX803_SGEMM_TILE + 1];
    __shared__ float Bs[GFX803_SGEMM_TILE][GFX803_SGEMM_TILE + 1];

    // Grid x walks row tiles for the same reason one level up: workgroups
    // dispatched together then write neighbouring rows of the same columns.
    // Walking column tiles instead put concurrent workgroups 16*ldc floats
    // apart (256 KiB at ldc 4096), all in the same channel.
    int tx = threadIdx.x, ty = threadIdx.y;
    int row0 = blockIdx.x * GFX803_SGEMM_TILE;
    int col0 = blockIdx.y * GFX803_SGEMM_TILE;
    int row = row0 + tx; // 0..M-1 (C row index)
    int col = col0 + ty; // 0..N-1 (C col index)

    float acc = 0.0f;
    int n_tiles = (K + GFX803_SGEMM_TILE - 1) / GFX803_SGEMM_TILE;

    for (int t = 0; t < n_tiles; t++) {
        int k0 = t * GFX803_SGEMM_TILE;

        // As[r][k] = op(A)(row0 + r, k0 + k)
        // transA=false: A physically MxK, op(A)(row,k) = A[k*lda + row]; rows contiguous
        // transA=true:  A physically KxM, op(A)(row,k) = A[row*lda + k]; k contiguous
        {
            int r = transA ? ty : tx;
            int k = transA ? tx : ty;
            int gr = row0 + r, gk = k0 + k;
            As[r][k] = (gr < M && gk < K)
                ? static_cast<float>(transA ? A[(size_t)gr * lda + gk] : A[(size_t)gk * lda + gr])
                : 0.0f;
        }

        // Bs[k][c] = op(B)(k0 + k, col0 + c)
        // transB=false: B physically KxN, op(B)(k,col) = B[col*ldb + k]; k contiguous
        // transB=true:  B physically NxK, op(B)(k,col) = B[k*ldb + col]; cols contiguous
        {
            int k = transB ? ty : tx;
            int c = transB ? tx : ty;
            int gk = k0 + k, gc = col0 + c;
            Bs[k][c] = (gc < N && gk < K)
                ? static_cast<float>(transB ? B[(size_t)gk * ldb + gc] : B[(size_t)gc * ldb + gk])
                : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GFX803_SGEMM_TILE; k++) {
            acc += As[tx][k] * Bs[k][ty];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        size_t idx = (size_t)col * ldc + row;
        C[idx] = static_cast<T>(alpha * acc + (beta != 0.0f ? beta * static_cast<float>(C[idx]) : 0.0f));
    }
}

#define GFX803_SGEMM_BIG_K 16

template <int P> struct gfx803_vec;
template <> struct gfx803_vec<2> {
    typedef float2 type;
    static __device__ __forceinline__ void unpack(type v, float* o) { o[0] = v.x; o[1] = v.y; }
};
template <> struct gfx803_vec<4> {
    typedef float4 type;
    static __device__ __forceinline__ void unpack(type v, float* o) {
        o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
    }
};

// Four or two consecutive operand values as one global load, widened to fp32.
// For fp16 that is one 8- or 4-byte load (flat_load_dwordx2/dword) and one
// v_cvt_f32_f16 per value, two of them SDWA on the high halves. gfx8 runs with
// SH_MEM_CONFIG in unaligned mode under amdkfd, and the compiler emits the
// same single load for a 2-byte-aligned address, so an odd K, leading
// dimension or offset needs no scalar path; edge tiles use the guarded scalar
// loads in the callers.
template <typename E> struct gfx803_ld;
template <> struct gfx803_ld<float> {
    static __device__ __forceinline__ void v4(const float* p, float4& o) { o = *reinterpret_cast<const float4*>(p); }
    static __device__ __forceinline__ void v2(const float* p, float2& o) { o = *reinterpret_cast<const float2*>(p); }
};
template <> struct gfx803_ld<_Float16> {
    typedef _Float16 h4 __attribute__((ext_vector_type(4)));
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    static __device__ __forceinline__ void v4(const _Float16* p, float4& o) {
        h4 v = *reinterpret_cast<const h4*>(p);
        o = make_float4(v.x, v.y, v.z, v.w);
    }
    static __device__ __forceinline__ void v2(const _Float16* p, float2& o) {
        h2 v = *reinterpret_cast<const h2*>(p);
        o = make_float2(v.x, v.y);
    }
};

// Same contract as gfx803_sgemm_tile, for a (16*P)x(16*P) tile of C: each of
// the 16x16 threads owns a PxP block of outputs (rows P*tx.., cols P*ty..), so
// every pair of values read from LDS feeds P*P multiply-adds instead of one.
// Each output still accumulates its K products one at a time in increasing k,
// so the result is bit-identical to gfx803_sgemm_tile's.
template <int P, typename E>
__device__ __forceinline__ void gfx803_sgemm_tile_p(int M, int N, int K,
                                                    float alpha,
                                                    const E* __restrict__ A, int lda, bool transA,
                                                    const E* __restrict__ B, int ldb, bool transB,
                                                    float beta,
                                                    E* __restrict__ C, int ldc) {
    constexpr int T = 16 * P, KT = GFX803_SGEMM_BIG_K;
    constexpr int LOADS = T * KT / 256;
    typedef typename gfx803_vec<P>::type vec;
    // k-major so a thread's P rows (or columns) for one k are one aligned
    // vector. The +4 keeps that alignment and spreads the transposed-operand
    // writes (16 k values of one row per 16 lanes) over more banks.
    __shared__ float As[KT][T + 4];
    __shared__ float Bs[KT][T + 4];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int row0 = blockIdx.x * T;
    const int col0 = blockIdx.y * T;

    // The 256 threads fetch each Tx16 operand slice so that consecutive
    // lanes read consecutive addresses whichever index is contiguous, for the
    // same DRAM-channel reason as in gfx803_sgemm_tile.
    float pa[LOADS], pb[LOADS];
    auto a_rk = [&](int lin, int& r, int& k) {
        if (transA) { r = lin / KT; k = lin % KT; } else { r = lin % T; k = lin / T; }
    };
    auto b_ck = [&](int lin, int& c, int& k) {
        if (transB) { c = lin % T; k = lin / T; } else { c = lin / KT; k = lin % KT; }
    };
    auto fetch = [&](int k0) {
        #pragma unroll
        for (int i = 0; i < LOADS; i++) {
            int r, c, k;
            a_rk(tid + 256 * i, r, k);
            int gr = row0 + r, gk = k0 + k;
            pa[i] = (gr < M && gk < K)
                ? static_cast<float>(transA ? A[(size_t)gr * lda + gk] : A[(size_t)gk * lda + gr])
                : 0.0f;
            b_ck(tid + 256 * i, c, k);
            int gc = col0 + c;
            gk = k0 + k;
            pb[i] = (gc < N && gk < K)
                ? static_cast<float>(transB ? B[(size_t)gk * ldb + gc] : B[(size_t)gc * ldb + gk])
                : 0.0f;
        }
    };

    float acc[P][P];
    #pragma unroll
    for (int i = 0; i < P; i++)
        #pragma unroll
        for (int j = 0; j < P; j++) acc[i][j] = 0.0f;

    const int n_tiles = (K + KT - 1) / KT;
    fetch(0);
    for (int t = 0; t < n_tiles; t++) {
        #pragma unroll
        for (int i = 0; i < LOADS; i++) {
            int r, c, k;
            a_rk(tid + 256 * i, r, k);
            As[k][r] = pa[i];
            b_ck(tid + 256 * i, c, k);
            Bs[k][c] = pb[i];
        }
        __syncthreads();
        // The next slice's global loads are issued before this slice's math
        // so their latency hides behind it.
        if (t + 1 < n_tiles) fetch((t + 1) * KT);

        float av[2][P], bv[2][P];
        gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&As[0][tx * P]), av[0]);
        gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&Bs[0][ty * P]), bv[0]);

        #pragma unroll
        for (int k = 0; k < KT; k++) {
            int cur = k & 1;
            int nxt = (k + 1) & 1;
            if (k + 1 < KT) {
                gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&As[k + 1][tx * P]), av[nxt]);
                gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&Bs[k + 1][ty * P]), bv[nxt]);
            }
            #pragma unroll
            for (int i = 0; i < P; i++)
                #pragma unroll
                for (int j = 0; j < P; j++) acc[i][j] += av[cur][i] * bv[cur][j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < P; j++) {
        int col = col0 + ty * P + j;
        #pragma unroll
        for (int i = 0; i < P; i++) {
            int row = row0 + tx * P + i;
            if (row < M && col < N) {
                size_t idx = (size_t)col * ldc + row;
                C[idx] = static_cast<E>(alpha * acc[i][j] + (beta != 0.0f ? beta * static_cast<float>(C[idx]) : 0.0f));
            }
        }
    }
}

template <typename T>
__global__ void gfx803_sgemm_kernel(int M, int N, int K,
                                    float alpha,
                                    const T* __restrict__ A, int lda, bool transA,
                                    const T* __restrict__ B, int ldb, bool transB,
                                    float beta,
                                    T* __restrict__ C, int ldc) {
    gfx803_sgemm_tile(M, N, K, alpha, A, lda, transA, B, ldb, transB, beta, C, ldc);
}

template <typename T>
__global__ void gfx803_sgemm_strided_batched_kernel(int M, int N, int K,
                                                    float alpha,
                                                    const T* __restrict__ A, int lda,
                                                    int64_t strideA, bool transA,
                                                    const T* __restrict__ B, int ldb,
                                                    int64_t strideB, bool transB,
                                                    float beta,
                                                    T* __restrict__ C, int ldc,
                                                    int64_t strideC) {
    const int64_t b = (int64_t)blockIdx.z;
    gfx803_sgemm_tile(M, N, K, alpha,
                      A + b * strideA, lda, transA,
                      B + b * strideB, ldb, transB,
                      beta,
                      C + b * strideC, ldc);
}

template <int P, typename T>
__global__ void __launch_bounds__(256)
gfx803_sgemm_p_kernel(int M, int N, int K,
                      float alpha,
                      const T* __restrict__ A, int lda, bool transA,
                      const T* __restrict__ B, int ldb, bool transB,
                      float beta,
                      T* __restrict__ C, int ldc) {
    gfx803_sgemm_tile_p<P, T>(M, N, K, alpha, A, lda, transA, B, ldb, transB, beta, C, ldc);
}

template <int P, typename T>
__global__ void __launch_bounds__(256)
gfx803_sgemm_p_strided_batched_kernel(int M, int N, int K,
                                      float alpha,
                                      const T* __restrict__ A, int lda,
                                      int64_t strideA, bool transA,
                                      const T* __restrict__ B, int ldb,
                                      int64_t strideB, bool transB,
                                      float beta,
                                      T* __restrict__ C, int ldc,
                                      int64_t strideC) {
    const int64_t b = (int64_t)blockIdx.z;
    gfx803_sgemm_tile_p<P, T>(M, N, K, alpha,
                           A + b * strideA, lda, transA,
                           B + b * strideB, ldb, transB,
                           beta,
                           C + b * strideC, ldc);
}

// gfx803_sgemm_tile_p spends part of every K step waiting: each thread reads
// its next A and B values from LDS and then has nothing to do until they
// arrive. Here the reads for step k+1 are issued before the multiply-adds of
// step k (av/bv double-buffered), so the LDS latency overlaps arithmetic.
// With A not transposed its column is contiguous along M, so a thread fetches
// four rows with one float4 global load and stores them with one LDS write.
// B is staged with consecutive lanes on consecutive K, its contiguous
// dimension when B is not transposed; with B transposed that dimension is N
// and the same mapping would stride every lane by ldb, so transposed-B calls
// keep gfx803_sgemm_tile_p. TransA is a template parameter so the fetch has
// no runtime branch on layout. The wide loads need no alignment (see
// gfx803_ld), and edge tiles fall back to guarded scalar loads.
template <int P, bool TransA, typename E, bool D = false>
__device__ __forceinline__ void gfx803_sgemm_tile_p_fast(int M, int N, int K,
                                                          float alpha,
                                                          const E* __restrict__ A, int lda,
                                                          const E* __restrict__ B, int ldb,
                                                          float beta,
                                                          E* __restrict__ C, int ldc) {
    constexpr int T = 16 * P, KT = GFX803_SGEMM_BIG_K;
    typedef typename gfx803_vec<P>::type vec;
    __shared__ float As[KT][T + 4];
    __shared__ float Bs[KT][T + 4];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int row0 = blockIdx.x * T;
    const int col0 = blockIdx.y * T;

    int k_a, r_a, gr_a;
    if constexpr (!TransA) {
        k_a = tid >> 4;
        r_a = (tid & 15) << 2;
        gr_a = row0 + r_a;
    } else {
        r_a = tid >> 2;
        k_a = (tid & 3) << 2;
        gr_a = row0 + r_a;
    }

    const int k_b = tid & 15;
    const int c_b_base = tid >> 4;

    float4 pa_vec;
    float pb[4];
    // Raw fp16 operands for D: see gfx803_stage_d for why the widening waits
    // for the LDS store.
    typedef E e4 __attribute__((ext_vector_type(4)));
    e4 pa_raw;
    E pb_raw[4];
    auto fetch_d = [&](int k0) {
        const int gk_a = k0 + k_a;
        if constexpr (!TransA) {
            const E* p = A + (size_t)gk_a * lda + gr_a;
            if (gr_a + 3 < M && gk_a < K) {
                pa_raw = *reinterpret_cast<const e4*>(p);
            } else {
                #pragma unroll
                for (int e = 0; e < 4; e++) pa_raw[e] = (gr_a + e < M && gk_a < K) ? p[e] : E(0);
            }
        } else {
            const E* p = A + (size_t)gr_a * lda + gk_a;
            if (gr_a < M && gk_a + 3 < K) {
                pa_raw = *reinterpret_cast<const e4*>(p);
            } else {
                #pragma unroll
                for (int e = 0; e < 4; e++) pa_raw[e] = (gr_a < M && gk_a + e < K) ? p[e] : E(0);
            }
        }
        const int gk = k0 + k_b;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int gc = col0 + c_b_base + 16 * i;
            pb_raw[i] = (gc < N && gk < K) ? B[(size_t)gc * ldb + gk] : E(0);
        }
    };

    auto fetch = [&](int k0) {
        int gk_a = k0 + k_a;
        if constexpr (!TransA) {
            if (gr_a + 3 < M && gk_a < K) {
                gfx803_ld<E>::v4(&A[(size_t)gk_a * lda + gr_a], pa_vec);
            } else {
                pa_vec = make_float4(
                    (gr_a + 0 < M && gk_a < K) ? static_cast<float>(A[(size_t)gk_a * lda + gr_a + 0]) : 0.0f,
                    (gr_a + 1 < M && gk_a < K) ? static_cast<float>(A[(size_t)gk_a * lda + gr_a + 1]) : 0.0f,
                    (gr_a + 2 < M && gk_a < K) ? static_cast<float>(A[(size_t)gk_a * lda + gr_a + 2]) : 0.0f,
                    (gr_a + 3 < M && gk_a < K) ? static_cast<float>(A[(size_t)gk_a * lda + gr_a + 3]) : 0.0f
                );
            }
        } else {
            if (gr_a < M && gk_a + 3 < K) {
                gfx803_ld<E>::v4(&A[(size_t)gr_a * lda + gk_a], pa_vec);
            } else {
                pa_vec = make_float4(
                    (gr_a < M && gk_a + 0 < K) ? static_cast<float>(A[(size_t)gr_a * lda + gk_a + 0]) : 0.0f,
                    (gr_a < M && gk_a + 1 < K) ? static_cast<float>(A[(size_t)gr_a * lda + gk_a + 1]) : 0.0f,
                    (gr_a < M && gk_a + 2 < K) ? static_cast<float>(A[(size_t)gr_a * lda + gk_a + 2]) : 0.0f,
                    (gr_a < M && gk_a + 3 < K) ? static_cast<float>(A[(size_t)gr_a * lda + gk_a + 3]) : 0.0f
                );
            }
        }

        int gk = k0 + k_b;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int gc = col0 + c_b_base + 16 * i;
            pb[i] = (gc < N && gk < K) ? static_cast<float>(B[(size_t)gc * ldb + gk]) : 0.0f;
        }
    };

    float acc[P][P];
    #pragma unroll
    for (int i = 0; i < P; i++)
        #pragma unroll
        for (int j = 0; j < P; j++) acc[i][j] = 0.0f;

    const int n_tiles = (K + KT - 1) / KT;
    auto fetch_any = [&](int k0) {
        if constexpr (D) fetch_d(k0); else fetch(k0);
    };
    fetch_any(0);
    for (int t = 0; t < n_tiles; t++) {
        if constexpr (D) {
            if constexpr (!TransA) {
                *reinterpret_cast<float4*>(&As[k_a][r_a]) =
                    make_float4(static_cast<float>(pa_raw[0]), static_cast<float>(pa_raw[1]),
                                static_cast<float>(pa_raw[2]), static_cast<float>(pa_raw[3]));
            } else {
                #pragma unroll
                for (int e = 0; e < 4; e++) As[k_a + e][r_a] = static_cast<float>(pa_raw[e]);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) Bs[k_b][c_b_base + 16 * i] = static_cast<float>(pb_raw[i]);
        } else {
            if constexpr (!TransA) {
                *reinterpret_cast<float4*>(&As[k_a][r_a]) = pa_vec;
            } else {
                As[k_a + 0][r_a] = pa_vec.x;
                As[k_a + 1][r_a] = pa_vec.y;
                As[k_a + 2][r_a] = pa_vec.z;
                As[k_a + 3][r_a] = pa_vec.w;
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                Bs[k_b][c_b_base + 16 * i] = pb[i];
            }
        }

        __syncthreads();
        if (t + 1 < n_tiles) fetch_any((t + 1) * KT);

        float av[2][P], bv[2][P];
        gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&As[0][tx * P]), av[0]);
        gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&Bs[0][ty * P]), bv[0]);

        #pragma unroll
        for (int k = 0; k < KT; k++) {
            int cur = k & 1;
            int nxt = (k + 1) & 1;
            if (k + 1 < KT) {
                gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&As[k + 1][tx * P]), av[nxt]);
                gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&Bs[k + 1][ty * P]), bv[nxt]);
            }
            #pragma unroll
            for (int i = 0; i < P; i++)
                #pragma unroll
                for (int j = 0; j < P; j++) acc[i][j] += av[cur][i] * bv[cur][j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < P; j++) {
        int col = col0 + ty * P + j;
        #pragma unroll
        for (int i = 0; i < P; i++) {
            int row = row0 + tx * P + i;
            if (row < M && col < N) {
                size_t idx = (size_t)col * ldc + row;
                C[idx] = static_cast<E>(alpha * acc[i][j] + (beta != 0.0f ? beta * static_cast<float>(C[idx]) : 0.0f));
            }
        }
    }
}

// The 32x32 tile with the scheme of gfx803_sgemm_tile_p_fast. Each thread
// fetches two values of A and two of B per K slice, so the wide loads are
// float2.
template <bool TransA, typename E, bool D = false>
__device__ __forceinline__ void gfx803_sgemm_tile_p2_fast(int M, int N, int K,
                                                          float alpha,
                                                          const E* __restrict__ A, int lda,
                                                          const E* __restrict__ B, int ldb,
                                                          float beta,
                                                          E* __restrict__ C, int ldc) {
    constexpr int P = 2;
    constexpr int T = 16 * P, KT = GFX803_SGEMM_BIG_K;
    typedef typename gfx803_vec<P>::type vec;
    __shared__ float As[KT][T + 4];
    __shared__ float Bs[KT][T + 4];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int row0 = blockIdx.x * T;
    const int col0 = blockIdx.y * T;

    const int r_a = (tid & 15) << 1;
    const int k_a = tid >> 4;
    const int gr_a = row0 + r_a;

    const int c_b = tid >> 3;
    const int k_b = (tid & 7) << 1;
    const int gc_b = col0 + c_b;

    float2 pa, pb;
    typedef E e2 __attribute__((ext_vector_type(2)));
    e2 pa_raw, pb_raw;
    auto fetch_d = [&](int k0) {
        const int gk_a = k0 + k_a;
        if constexpr (!TransA) {
            const E* p = A + (size_t)gk_a * lda + gr_a;
            if (gr_a + 1 < M && gk_a < K) {
                pa_raw = *reinterpret_cast<const e2*>(p);
            } else {
                #pragma unroll
                for (int e = 0; e < 2; e++) pa_raw[e] = (gr_a + e < M && gk_a < K) ? p[e] : E(0);
            }
        } else {
            #pragma unroll
            for (int e = 0; e < 2; e++)
                pa_raw[e] = (gr_a + e < M && gk_a < K) ? A[(size_t)(gr_a + e) * lda + gk_a] : E(0);
        }
        const int gk_b = k0 + k_b;
        const E* q = B + (size_t)gc_b * ldb + gk_b;
        if (gc_b < N && gk_b + 1 < K) {
            pb_raw = *reinterpret_cast<const e2*>(q);
        } else {
            #pragma unroll
            for (int e = 0; e < 2; e++) pb_raw[e] = (gc_b < N && gk_b + e < K) ? q[e] : E(0);
        }
    };

    auto fetch = [&](int k0) {
        int gk_a = k0 + k_a;
        if constexpr (!TransA) {
            if (gr_a + 1 < M && gk_a < K) {
                gfx803_ld<E>::v2(&A[(size_t)gk_a * lda + gr_a], pa);
            } else {
                pa = make_float2(
                    (gr_a + 0 < M && gk_a < K) ? static_cast<float>(A[(size_t)gk_a * lda + gr_a + 0]) : 0.0f,
                    (gr_a + 1 < M && gk_a < K) ? static_cast<float>(A[(size_t)gk_a * lda + gr_a + 1]) : 0.0f
                );
            }
        } else {
            pa = make_float2(
                (gr_a + 0 < M && gk_a < K) ? static_cast<float>(A[(size_t)(gr_a + 0) * lda + gk_a]) : 0.0f,
                (gr_a + 1 < M && gk_a < K) ? static_cast<float>(A[(size_t)(gr_a + 1) * lda + gk_a]) : 0.0f
            );
        }

        int gk_b = k0 + k_b;
        if (gc_b < N && gk_b + 1 < K) {
            gfx803_ld<E>::v2(&B[(size_t)gc_b * ldb + gk_b], pb);
        } else {
            pb = make_float2(
                (gc_b < N && gk_b + 0 < K) ? static_cast<float>(B[(size_t)gc_b * ldb + gk_b + 0]) : 0.0f,
                (gc_b < N && gk_b + 1 < K) ? static_cast<float>(B[(size_t)gc_b * ldb + gk_b + 1]) : 0.0f
            );
        }
    };

    float acc[P][P];
    #pragma unroll
    for (int i = 0; i < P; i++)
        #pragma unroll
        for (int j = 0; j < P; j++) acc[i][j] = 0.0f;

    const int n_tiles = (K + KT - 1) / KT;
    auto fetch_any = [&](int k0) {
        if constexpr (D) fetch_d(k0); else fetch(k0);
    };
    fetch_any(0);
    for (int t = 0; t < n_tiles; t++) {
        if constexpr (D) {
            *reinterpret_cast<float2*>(&As[k_a][r_a]) =
                make_float2(static_cast<float>(pa_raw[0]), static_cast<float>(pa_raw[1]));
            Bs[k_b + 0][c_b] = static_cast<float>(pb_raw[0]);
            Bs[k_b + 1][c_b] = static_cast<float>(pb_raw[1]);
        } else {
            *reinterpret_cast<float2*>(&As[k_a][r_a]) = pa;

            Bs[k_b + 0][c_b] = pb.x;
            Bs[k_b + 1][c_b] = pb.y;
        }

        __syncthreads();
        if (t + 1 < n_tiles) fetch_any((t + 1) * KT);

        float av[2][P], bv[2][P];
        gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&As[0][tx * P]), av[0]);
        gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&Bs[0][ty * P]), bv[0]);

        #pragma unroll
        for (int k = 0; k < KT; k++) {
            int cur = k & 1;
            int nxt = (k + 1) & 1;
            if (k + 1 < KT) {
                gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&As[k + 1][tx * P]), av[nxt]);
                gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&Bs[k + 1][ty * P]), bv[nxt]);
            }
            #pragma unroll
            for (int i = 0; i < P; i++)
                #pragma unroll
                for (int j = 0; j < P; j++) acc[i][j] += av[cur][i] * bv[cur][j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < P; j++) {
        int col = col0 + ty * P + j;
        #pragma unroll
        for (int i = 0; i < P; i++) {
            int row = row0 + tx * P + i;
            if (row < M && col < N) {
                size_t idx = (size_t)col * ldc + row;
                C[idx] = static_cast<E>(alpha * acc[i][j] + (beta != 0.0f ? beta * static_cast<float>(C[idx]) : 0.0f));
            }
        }
    }
}

template <int P, bool TransA, typename T, bool D = false>
__global__ void __launch_bounds__(256)
gfx803_sgemm_p_fast_kernel(int M, int N, int K,
                           float alpha,
                           const T* __restrict__ A, int lda,
                           const T* __restrict__ B, int ldb,
                           float beta,
                           T* __restrict__ C, int ldc) {
    gfx803_sgemm_tile_p_fast<P, TransA, T, D>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <int P, bool TransA, typename T, bool D = false>
__global__ void __launch_bounds__(256)
gfx803_sgemm_p_strided_batched_fast_kernel(int M, int N, int K,
                                           float alpha,
                                           const T* __restrict__ A, int lda,
                                           int64_t strideA,
                                           const T* __restrict__ B, int ldb,
                                           int64_t strideB,
                                           float beta,
                                           T* __restrict__ C, int ldc,
                                           int64_t strideC) {
    const int64_t b = (int64_t)blockIdx.z;
    gfx803_sgemm_tile_p_fast<P, TransA, T, D>(M, N, K, alpha,
                                             A + b * strideA, lda,
                                             B + b * strideB, ldb,
                                             beta,
                                             C + b * strideC, ldc);
}

template <bool TransA, typename T, bool D = false>
__global__ void __launch_bounds__(256)
gfx803_sgemm_p2_fast_kernel(int M, int N, int K,
                            float alpha,
                            const T* __restrict__ A, int lda,
                            const T* __restrict__ B, int ldb,
                            float beta,
                            T* __restrict__ C, int ldc) {
    gfx803_sgemm_tile_p2_fast<TransA, T, D>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <bool TransA, typename T, bool D = false>
__global__ void __launch_bounds__(256)
gfx803_sgemm_p2_strided_batched_fast_kernel(int M, int N, int K,
                                            float alpha,
                                            const T* __restrict__ A, int lda,
                                            int64_t strideA,
                                            const T* __restrict__ B, int ldb,
                                            int64_t strideB,
                                            float beta,
                                            T* __restrict__ C, int ldc,
                                            int64_t strideC) {
    const int64_t b = (int64_t)blockIdx.z;
    gfx803_sgemm_tile_p2_fast<TransA, T, D>(M, N, K, alpha,
                                            A + b * strideA, lda,
                                            B + b * strideB, ldb,
                                            beta,
                                            C + b * strideC, ldc);
}

// One operand's Kx(16P) slice for gfx803_sgemm_tile_g, staged into LDS as
// Xs[k][m] (k-major, as the multiply-add loop reads it). KC: the operand's
// contiguous index is K (A transposed, B not transposed); otherwise it is the
// M or N index and a thread moves four rows with one float4 load and one LDS
// write. A K-contiguous operand is read either as float4 runs along K, with
// consecutive lanes on consecutive addresses of one row (Scalar = false), or
// as single values with 16 lanes along K (Scalar = true); which is faster
// depends on the shape, see gfx803_sgemm_tile_g. Edge slices fall back to
// guarded scalar loads. E is the storage type (float or _Float16); the LDS
// slice is fp32 either way.
template <int TT, int KT, bool KC, bool Scalar, typename E>
struct gfx803_stage {
    static constexpr int NV = TT * KT / 4 / 256;
    static constexpr int NS = TT * KT / 256;
    static constexpr int KQ = KT / 4;
    static_assert(NV >= 1 && TT * KT % 1024 == 0, "slice must give every thread whole float4s");
    float4 v[NV];
    float s[NS];

    __device__ __forceinline__ void fetch(const E* __restrict__ X, int ld, int rows, int K,
                                          int idx0, int k0, int tid) {
        auto at = [](const E* p, bool ok) { return ok ? static_cast<float>(*p) : 0.0f; };
        if constexpr (KC && Scalar) {
            #pragma unroll
            for (int i = 0; i < NS; i++) {
                int lin = tid + 256 * i;
                int gr = idx0 + lin / KT, gk = k0 + lin % KT;
                s[i] = at(X + (size_t)gr * ld + gk, gr < rows && gk < K);
            }
        } else if constexpr (KC) {
            #pragma unroll
            for (int i = 0; i < NV; i++) {
                int lin = tid + 256 * i;
                int gr = idx0 + lin / KQ, gk = k0 + (lin % KQ) * 4;
                const E* p = X + (size_t)gr * ld + gk;
                if (gr < rows && gk + 3 < K) gfx803_ld<E>::v4(p, v[i]);
                else v[i] = make_float4(at(p + 0, gr < rows && gk + 0 < K), at(p + 1, gr < rows && gk + 1 < K),
                                        at(p + 2, gr < rows && gk + 2 < K), at(p + 3, gr < rows && gk + 3 < K));
            }
        } else {
            #pragma unroll
            for (int i = 0; i < NV; i++) {
                int lin = tid + 256 * i;
                int gr = idx0 + (lin % (TT / 4)) * 4, gk = k0 + lin / (TT / 4);
                const E* p = X + (size_t)gk * ld + gr;
                if (gr + 3 < rows && gk < K) gfx803_ld<E>::v4(p, v[i]);
                else v[i] = make_float4(at(p + 0, gr + 0 < rows && gk < K), at(p + 1, gr + 1 < rows && gk < K),
                                        at(p + 2, gr + 2 < rows && gk < K), at(p + 3, gr + 3 < rows && gk < K));
            }
        }
    }

    template <int S>
    __device__ __forceinline__ void store(float (*Xs)[S], int tid) const {
        if constexpr (KC && Scalar) {
            #pragma unroll
            for (int i = 0; i < NS; i++) {
                int lin = tid + 256 * i;
                Xs[lin % KT][lin / KT] = s[i];
            }
        } else if constexpr (KC) {
            #pragma unroll
            for (int i = 0; i < NV; i++) {
                int lin = tid + 256 * i, r = lin / KQ, k = (lin % KQ) * 4;
                Xs[k + 0][r] = v[i].x;
                Xs[k + 1][r] = v[i].y;
                Xs[k + 2][r] = v[i].z;
                Xs[k + 3][r] = v[i].w;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < NV; i++) {
                int lin = tid + 256 * i;
                *reinterpret_cast<float4*>(&Xs[lin / (TT / 4)][(lin % (TT / 4)) * 4]) = v[i];
            }
        }
    }
};

// gfx803_stage with the fetched values kept in storage type E until the LDS
// store widens them. Widened right after the load, an fp16 value makes the
// compiler wait for that load (s_waitcnt vmcnt(0)) inside the guarded fetch,
// one load at a time, before the slice's multiply-adds start: the fp16 tiles
// then pay four or five memory round trips per K slice that nothing in the
// same wavefront overlaps. Kept raw, a value is first needed at the next
// slice's LDS store, so the loads stay in flight during the multiply-adds, as
// they do in the fp32 kernels. The same fp32 values reach LDS, so results are
// bit-identical.
template <int TT, int KT, bool KC, bool Scalar, typename E>
struct gfx803_stage_d {
    static constexpr int NV = TT * KT / 4 / 256;
    static constexpr int NS = TT * KT / 256;
    static constexpr int KQ = KT / 4;
    static_assert(NV >= 1 && TT * KT % 1024 == 0, "slice must give every thread whole float4s");
    typedef E e4 __attribute__((ext_vector_type(4)));
    e4 v[NV];
    E s[NS];

    __device__ __forceinline__ void fetch(const E* __restrict__ X, int ld, int rows, int K,
                                          int idx0, int k0, int tid) {
        auto at = [](const E* p, bool ok) { return ok ? *p : E(0); };
        if constexpr (KC && Scalar) {
            #pragma unroll
            for (int i = 0; i < NS; i++) {
                int lin = tid + 256 * i;
                int gr = idx0 + lin / KT, gk = k0 + lin % KT;
                s[i] = at(X + (size_t)gr * ld + gk, gr < rows && gk < K);
            }
        } else if constexpr (KC) {
            #pragma unroll
            for (int i = 0; i < NV; i++) {
                int lin = tid + 256 * i;
                int gr = idx0 + lin / KQ, gk = k0 + (lin % KQ) * 4;
                const E* p = X + (size_t)gr * ld + gk;
                if (gr < rows && gk + 3 < K) {
                    v[i] = *reinterpret_cast<const e4*>(p);
                } else {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) v[i][e] = at(p + e, gr < rows && gk + e < K);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < NV; i++) {
                int lin = tid + 256 * i;
                int gr = idx0 + (lin % (TT / 4)) * 4, gk = k0 + lin / (TT / 4);
                const E* p = X + (size_t)gk * ld + gr;
                if (gr + 3 < rows && gk < K) {
                    v[i] = *reinterpret_cast<const e4*>(p);
                } else {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) v[i][e] = at(p + e, gr + e < rows && gk < K);
                }
            }
        }
    }

    template <int S>
    __device__ __forceinline__ void store(float (*Xs)[S], int tid) const {
        if constexpr (KC && Scalar) {
            #pragma unroll
            for (int i = 0; i < NS; i++) {
                int lin = tid + 256 * i;
                Xs[lin % KT][lin / KT] = static_cast<float>(s[i]);
            }
        } else if constexpr (KC) {
            #pragma unroll
            for (int i = 0; i < NV; i++) {
                int lin = tid + 256 * i, r = lin / KQ, k = (lin % KQ) * 4;
                #pragma unroll
                for (int e = 0; e < 4; e++) Xs[k + e][r] = static_cast<float>(v[i][e]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < NV; i++) {
                int lin = tid + 256 * i;
                *reinterpret_cast<float4*>(&Xs[lin / (TT / 4)][(lin % (TT / 4)) * 4]) =
                    make_float4(static_cast<float>(v[i][0]), static_cast<float>(v[i][1]),
                                static_cast<float>(v[i][2]), static_cast<float>(v[i][3]));
            }
        }
    }
};
// gfx803_sgemm_tile_p_fast for the transposed layouts. nn.Linear reaches
// rocBLAS as op(A) = T, op(B) = N (40% of an SD1.5 step's GEMM FLOPs) and its
// weight gradient as op(A) = N, op(B) = T; gfx803_sgemm_tile_p_fast stages a
// transposed B with every lane strided by ldb, and its transposed-A fetch
// reads each row of A as one 64-byte piece. Measured on an RX 590 against
// gfx803_sgemm_tile_p_fast / gfx803_sgemm_tile_p, both kernels' outputs
// bit-identical:
//   - op(B) = T: B is staged like A in the NN kernel, four N-contiguous
//     values per load. Weight gradients of SD1.5's linears go from 2.5-3.2 to
//     3.8-5.2 TFLOPS (640x5120x2048: 3.0 -> 5.2).
//   - op(A) = T with an A larger than the 2 MiB L2 (feed-forward and other
//     wide linears): a 32-deep K slice, so each row of A is read in 128-byte
//     pieces, and the tiles go out in groups of GW column tiles that walk the
//     row tiles together (blockIdx.x is a linear tile index). Row tiles alone
//     re-read all of A from DRAM for every column tile; a group reads it once
//     per GW column tiles. 2560x8192x320: 3.2 -> 4.4; 5120x2048x640:
//     2.8 -> 4.4. With K >= 2048 a K-contiguous B is faster as single floats
//     (640x2048x2560: 2.2 -> 3.6; 1280x512x5120: 2.0 -> 3.2).
// Every output still accumulates its products one at a time in increasing k,
// so the result is bit-identical to the other tiles. See gfx803_sgemm_g_launch
// for the shapes that keep the older kernels.
template <int P, int KT, bool TransA, bool TransB, bool ScalarB, typename E, bool D = false>
__device__ __forceinline__ void gfx803_sgemm_tile_g(int GW, int M, int N, int K, float alpha,
                                                    const E* __restrict__ A, int lda,
                                                    const E* __restrict__ B, int ldb,
                                                    float beta, E* __restrict__ C, int ldc) {
    constexpr int T = 16 * P;
    typedef typename gfx803_vec<P>::type vec;
    __shared__ float As[KT][T + 4];
    __shared__ float Bs[KT][T + 4];
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;

    const int gx = (M + T - 1) / T, gy = (N + T - 1) / T;
    const int per = GW * gx, grp = blockIdx.x / per, first = grp * GW;
    const int gw = gy - first < GW ? gy - first : GW, loc = blockIdx.x - grp * per;
    const int row0 = (loc / gw) * T, col0 = (first + loc % gw) * T;

    std::conditional_t<D, gfx803_stage_d<T, KT, TransA, false, E>, gfx803_stage<T, KT, TransA, false, E>> sa;
    std::conditional_t<D, gfx803_stage_d<T, KT, !TransB, ScalarB, E>, gfx803_stage<T, KT, !TransB, ScalarB, E>> sb;

    float acc[P][P];
    #pragma unroll
    for (int i = 0; i < P; i++)
        #pragma unroll
        for (int j = 0; j < P; j++) acc[i][j] = 0.0f;

    const int n_tiles = (K + KT - 1) / KT;
    sa.fetch(A, lda, M, K, row0, 0, tid);
    sb.fetch(B, ldb, N, K, col0, 0, tid);
    for (int t = 0; t < n_tiles; t++) {
        sa.store(As, tid);
        sb.store(Bs, tid);
        __syncthreads();
        if (t + 1 < n_tiles) {
            sa.fetch(A, lda, M, K, row0, (t + 1) * KT, tid);
            sb.fetch(B, ldb, N, K, col0, (t + 1) * KT, tid);
        }

        float av[2][P], bv[2][P];
        gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&As[0][tx * P]), av[0]);
        gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&Bs[0][ty * P]), bv[0]);
        #pragma unroll
        for (int k = 0; k < KT; k++) {
            int cur = k & 1, nxt = (k + 1) & 1;
            if (k + 1 < KT) {
                gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&As[k + 1][tx * P]), av[nxt]);
                gfx803_vec<P>::unpack(*reinterpret_cast<const vec*>(&Bs[k + 1][ty * P]), bv[nxt]);
            }
            #pragma unroll
            for (int i = 0; i < P; i++)
                #pragma unroll
                for (int j = 0; j < P; j++) acc[i][j] += av[cur][i] * bv[cur][j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < P; j++) {
        int col = col0 + ty * P + j;
        #pragma unroll
        for (int i = 0; i < P; i++) {
            int row = row0 + tx * P + i;
            if (row < M && col < N) {
                size_t idx = (size_t)col * ldc + row;
                C[idx] = static_cast<E>(alpha * acc[i][j] + (beta != 0.0f ? beta * static_cast<float>(C[idx]) : 0.0f));
            }
        }
    }
}

template <int P, int KT, bool TransA, bool TransB, bool ScalarB, typename E, bool D = false>
__global__ void __launch_bounds__(256)
gfx803_sgemm_g_kernel(int GW, int M, int N, int K, float alpha,
                      const E* __restrict__ A, int lda, int64_t strideA,
                      const E* __restrict__ B, int ldb, int64_t strideB,
                      float beta, E* __restrict__ C, int ldc, int64_t strideC) {
    const int64_t b = (int64_t)blockIdx.z;
    gfx803_sgemm_tile_g<P, KT, TransA, TransB, ScalarB, E, D>(GW, M, N, K, alpha, A + b * strideA, lda,
                                                              B + b * strideB, ldb, beta, C + b * strideC, ldc);
}

// Launches gfx803_sgemm_tile_g where it measured faster; returns false to
// leave the call to the other kernels. A leading dimension that is a multiple
// of 8 KiB puts every row of a K-contiguous operand in the same DRAM channel,
// and no tile order spreads that out, since one tile already reads 64 such
// rows at once. gfx803_sgemm_tile_g loses more there than the older kernels:
// 2048^3 with A transposed ran at 2.9 TFLOPS against 3.4 on
// gfx803_sgemm_tile_p_fast, with B transposed at 1.4 against 3.2, and
// 4096 x 512 x 4096 TN with only B at such a stride at 2.5 against 4.5. Those
// shapes keep the older kernels; padding the leading dimension (2048 -> 2064)
// takes them to the speed of the rest. Transposed-A GEMMs with A within L2,
// and K <= 64 where a 32-deep slice is mostly padding, measured the same or
// slower here than on gfx803_sgemm_tile_p_fast.
//
// Which staging of B and which tile order win for op(A) = T depends on how
// B's rows are aligned. With rows at multiples of 256 bytes, as nn.Linear's
// widths give, single-float B for K >= 2048 and grouped tiles measured best
// (640x2048x2560: 2.5 -> 3.6 TFLOPS against float4 B in row-tile order).
// Rows padded off that alignment, as llama.cpp's prompt GEMMs on gfx803 are
// (K + 16 floats), measured best with float4 B in row-tile order
// (8192x2048x2048: 2.1 -> 3.9; 4096x512x14336: 3.7 -> 3.9).
// The thresholds are in bytes, so for fp16 operands they fall at twice the
// element counts (an ld of 4096 halves camps; rows align at 128 halves); they
// were measured with fp32 operands. Single-float B is fp32-only: at fp16 each
// lane loads 2 bytes, and float4 B with grouped tiles measured faster at every
// fp16 shape where the choice applies (1280x512x5120: 2.1 -> 3.5 TFLOPS;
// 640x2048x2560: 2.8 -> 4.3, SD1.5's feed-forward output projections).
template <typename E>
inline bool gfx803_sgemm_g_applies(int tile, int M, int K, int lda, bool transA, int ldb, bool transB) {
    auto camps = [](int ld) { return ((size_t)ld * sizeof(E)) % 8192 == 0; };
    const bool a_big = (size_t)M * K * sizeof(E) > (2u << 20);
    if (transB ? (camps(lda) && camps(ldb)) : (!transA || K <= 64 || !a_big || camps(lda) || camps(ldb)))
        return false;
    return !(tile == 16 || (tile == 32 && !transB));
}

template <typename E, bool D = false>
inline bool gfx803_sgemm_g_launch(int tile, dim3 grid, hipStream_t stream, int M, int N, int K, float alpha,
                                  const E* A, int lda, int64_t strideA, bool transA,
                                  const E* B, int ldb, int64_t strideB, bool transB,
                                  float beta, E* C, int ldc, int64_t strideC) {
    if (!gfx803_sgemm_g_applies<E>(tile, M, K, lda, transA, ldb, transB)) return false;
    const bool a_big = (size_t)M * K * sizeof(E) > (2u << 20);
    const dim3 g1(grid.x * grid.y, 1, grid.z), block(GFX803_SGEMM_TILE, GFX803_SGEMM_TILE);
    const bool b_rows_aligned = transB || ((size_t)ldb * sizeof(E)) % 256 == 0;
    const int gw = transA && a_big && b_rows_aligned ? 8 : 1;
#define GFX803_G(P, TA, TB, SB) \
    hipLaunchKernelGGL((gfx803_sgemm_g_kernel<P, 32, TA, TB, SB, E, D>), g1, block, 0, stream, gw, M, N, K, alpha, \
                       A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC)
    if (tile == 32) {
        if (transA) GFX803_G(2, true, true, false); else GFX803_G(2, false, true, false);
    } else if (transB) {
        if (transA) GFX803_G(4, true, true, false); else GFX803_G(4, false, true, false);
    } else if (K >= 2048 && b_rows_aligned && sizeof(E) == sizeof(float)) {
        GFX803_G(4, true, false, true);
    } else {
        GFX803_G(4, true, false, false);
    }
#undef GFX803_G
    return true;
}

static inline long gfx803_sgemm_env_long(const char* name, long fallback) {
    const char* v = getenv(name);
    return v ? atol(v) : fallback;
}

// GEMMs with few columns of C and op(A) = T: a weight matrix times the
// one or two activation vectors of a token-by-token decode step (nn.Linear at
// batch 1-8, C = W^T-stored weight times x). The tiles above leave most of a
// 16-, 32- or 64-wide tile idle there, and rocBLAS's fp16 kernel for this
// layout uses a 128x32 tile: a 1024x2x1024 fp16 GEMM took 249 us where
// reading the weight takes about 10. Here each wavefront owns R rows of C (R
// rows of op(A), contiguous in K) or a share of their K, its lanes stride K
// with 16-byte loads, all columns' sums stay in registers, and a butterfly
// reduction ends the row, as a GEMV does. The R rows of a wavefront share
// each load of B. The few-column GEMMs of the token model in Chatterbox TTS
// (a 0.5B Llama run at batch 2) take 7 ms per token in fp16 here, 75.6 ms on
// rocBLAS; 10 ms in fp32, 43 ms on the tiles.
//
// Products are summed in a different order than in the tiles, so results
// differ from theirs in the last bits; the choice depends only on the shape,
// so a given call is deterministic.
template <typename T> struct gfx803_vec16;
template <> struct gfx803_vec16<float> {
    typedef float type __attribute__((ext_vector_type(4)));
    static constexpr int n = 4;
};
template <> struct gfx803_vec16<_Float16> {
    typedef _Float16 type __attribute__((ext_vector_type(8)));
    static constexpr int n = 8;
};

#define GFX803_SMALL_N_MAX 8

// Vec: A's rows and B's columns (op(B) = N) start 16-byte aligned, so K is
// read in 16-byte vectors up to the last whole vector. Without it, and for
// op(B) = T, every element is a scalar load, still contiguous across lanes
// for A.
template <int NC, int R, bool Vec, bool TransB, typename T>
__global__ void __launch_bounds__(256)
gfx803_sgemm_small_n_kernel(int M, int K, int S, float alpha,
                            const T* __restrict__ A, int lda, int64_t strideA,
                            const T* __restrict__ B, int ldb, int64_t strideB,
                            float beta, T* __restrict__ C, int ldc, int64_t strideC) {
    __shared__ float part[4][R * NC];
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int kpart = wave % S;
    const int m0 = (blockIdx.x * (4 / S) + wave / S) * R;
    const bool live = m0 < M;
    A += blockIdx.z * strideA;
    B += blockIdx.z * strideB;
    C += blockIdx.z * strideC;
    const T* Ar[R];
    #pragma unroll
    for (int r = 0; r < R; r++) Ar[r] = A + (size_t)(m0 + r < M ? m0 + r : (live ? m0 : 0)) * lda;

    float acc[R][NC];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < NC; j++) acc[r][j] = 0.0f;

    int k_tail = 0;
    if constexpr (Vec && !TransB) {
        using V = typename gfx803_vec16<T>::type;
        constexpr int W = gfx803_vec16<T>::n;
        k_tail = K / W * W;
        for (int k = (kpart * 64 + lane) * W; live && k < k_tail; k += 64 * W * S) {
            V a[R];
            #pragma unroll
            for (int r = 0; r < R; r++) a[r] = *(const V*)(Ar[r] + k);
            #pragma unroll
            for (int j = 0; j < NC; j++) {
                V b = *(const V*)(B + (size_t)j * ldb + k);
                #pragma unroll
                for (int e = 0; e < W; e++) {
                    const float be = (float)b[e];
                    #pragma unroll
                    for (int r = 0; r < R; r++) acc[r][j] += (float)a[r][e] * be;
                }
            }
        }
    }
    for (int k = k_tail + kpart * 64 + lane; live && k < K; k += 64 * S) {
        float a[R];
        #pragma unroll
        for (int r = 0; r < R; r++) a[r] = (float)Ar[r][k];
        #pragma unroll
        for (int j = 0; j < NC; j++) {
            const float b = (float)(TransB ? B[(size_t)k * ldb + j] : B[(size_t)j * ldb + k]);
            #pragma unroll
            for (int r = 0; r < R; r++) acc[r][j] += a[r] * b;
        }
    }

    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        #pragma unroll
        for (int r = 0; r < R; r++)
            #pragma unroll
            for (int j = 0; j < NC; j++) acc[r][j] += __shfl_xor(acc[r][j], off, 64);

    // Every lane now holds every sum of its wavefront; lane r*NC + j takes
    // C(m0 + r, j). Selecting with compares keeps acc in registers (a dynamic
    // index would put it in scratch memory).
    float v = 0.0f;
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < NC; j++)
            if (lane == r * NC + j) v = acc[r][j];
    if (S > 1) {
        if (lane < R * NC) part[wave][lane] = v;
        __syncthreads();
        if (kpart != 0) return;
        for (int p = 1; p < S; p++)
            if (lane < R * NC) v += part[wave + p][lane];
    }
    const int r = lane / NC, j = lane % NC;
    if (live && lane < R * NC && m0 + r < M) {
        size_t idx = (size_t)j * ldc + m0 + r;
        C[idx] = static_cast<T>(alpha * v + (beta != 0.0f ? beta * static_cast<float>(C[idx]) : 0.0f));
    }
}

// Two rows per wavefront halve the reads of B. A row of K is split over S
// wavefronts of the workgroup (summed through LDS) once each still gets two
// 16-byte loads per lane: a 1024x4096 weight otherwise leaves 3-4 wavefronts
// per SIMD, too few to hide memory latency (fp32 4096->1024 at batch 2:
// 270 -> 82 us with S = 4).
template <int NC, typename T>
inline void gfx803_small_n_launch(bool vec, bool transB, unsigned batch, hipStream_t stream,
                                  int M, int K, float alpha, const T* A, int lda, int64_t strideA,
                                  const T* B, int ldb, int64_t strideB, float beta, T* C, int ldc,
                                  int64_t strideC) {
    constexpr int R = 2;
    constexpr int W = gfx803_vec16<T>::n;
    const int S = K >= 4 * 64 * W * 2 ? 4 : K >= 2 * 64 * W * 2 ? 2 : 1;
    const int rows = 4 / S * R;
    dim3 grid((unsigned)((M + rows - 1) / rows), 1, batch);
    if (transB)
        hipLaunchKernelGGL((gfx803_sgemm_small_n_kernel<NC, R, false, true, T>), grid, dim3(256), 0, stream,
                           M, K, S, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC);
    else if (vec)
        hipLaunchKernelGGL((gfx803_sgemm_small_n_kernel<NC, R, true, false, T>), grid, dim3(256), 0, stream,
                           M, K, S, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC);
    else
        hipLaunchKernelGGL((gfx803_sgemm_small_n_kernel<NC, R, false, false, T>), grid, dim3(256), 0, stream,
                           M, K, S, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC);
}

// GFX803_SGEMM_SMALL_N sets the largest N this path takes (default 8, 0 = off),
// for benchmarking against the tiles.
inline bool gfx803_small_n_applies(int N, bool transA) {
    static const long max_n = gfx803_sgemm_env_long("GFX803_SGEMM_SMALL_N", GFX803_SMALL_N_MAX);
    return transA && N >= 1 && N <= max_n && N <= GFX803_SMALL_N_MAX;
}

template <typename T>
inline void gfx803_small_n_dispatch(int N, unsigned batch, hipStream_t stream, int M, int K, float alpha,
                                    const T* A, int lda, int64_t strideA, bool transB,
                                    const T* B, int ldb, int64_t strideB, float beta,
                                    T* C, int ldc, int64_t strideC) {
    constexpr int W = gfx803_vec16<T>::n;
    auto aligned = [](const void* p) { return ((uintptr_t)p & 15) == 0; };
    const bool vec = aligned(A) && aligned(B) && lda % W == 0 && ldb % W == 0 &&
                     (batch == 1 || (strideA % W == 0 && strideB % W == 0));
#define GFX803_SN(n) case n: gfx803_small_n_launch<n, T>(vec, transB, batch, stream, M, K, alpha, A, lda, \
        strideA, B, ldb, strideB, beta, C, ldc, strideC); break;
    switch (N) {
        GFX803_SN(1) GFX803_SN(2) GFX803_SN(3) GFX803_SN(4)
        GFX803_SN(5) GFX803_SN(6) GFX803_SN(7) GFX803_SN(8)
    }
#undef GFX803_SN
}

// Returns the C tile edge to use: 64, 32 or 16. All three give bit-identical
// results, so this choice is only about speed, and the rule comes from timing
// the shapes SD1.5 LoRA training issues on an RX 590:
//   - A larger tile needs about 1.5 tiles per CU; below that the smaller
//     tile's 4x more workgroups win (m=128 n=1280 k=11520, 40 64x64 tiles:
//     2.61 ms on 64x64, 2.43 ms on 32x32; m=192, 60 tiles: 2.70 vs 3.28 ms).
//   - With K <= 32 the 64x64 tile loses at any size, since the per-tile
//     fetch and the zero-padded K slice are not paid back (LoRA's k=8
//     m=4096 n=1280: 181 vs 128 us). From K = 40 it wins again (attention
//     scores 8x4096x40x4096: 6.6 vs 7.9 ms).
//   - An M or N of 16 or less leaves most of a 64-wide tile idle (m=8
//     n=4096 k=320: 80 us on 64x64, 59 us on 32x32); 40 still fills enough
//     of it to win (attention context, m=40: 8.7 vs 9.2 ms).
//   - With fp16 operands an M of 32 already loses on the 64x64 tile: the
//     Winograd solver's GEMMs at SD1.5's smallest latent (16 x 32x1280x1280:
//     0.86 vs 0.63 ms; k=2560: 2.33 vs 1.46 ms). With fp32 operands, and at
//     M = 48 in either type, the 64x64 tile stays faster.
// GFX803_SGEMM_MIN_TILES64 and GFX803_SGEMM_MIN_TILES32 override the tile-
// count thresholds for benchmarking (0 = always, -1 = never use that tile).
inline int gfx803_sgemm_pick_tile(int M, int N, int K, int batch, bool half = false) {
    static const long cus = [] {
        int dev = 0, n = 0;
        if (hipGetDevice(&dev) != hipSuccess ||
            hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess ||
            n <= 0)
            n = 36;
        return (long)n;
    }();
    static const long min64 = gfx803_sgemm_env_long("GFX803_SGEMM_MIN_TILES64", 3 * cus / 2);
    static const long min32 = gfx803_sgemm_env_long("GFX803_SGEMM_MIN_TILES32", 3 * cus / 2);
    static const bool min64_forced = getenv("GFX803_SGEMM_MIN_TILES64") != nullptr;
    auto tiles = [&](int t) {
        return (long)((M + t - 1) / t) * ((N + t - 1) / t) * batch;
    };
    const int min_m64 = half ? 32 : 16;
    if (min64 >= 0 && ((K > 32 && M > min_m64 && N > 16) || min64_forced) && tiles(64) >= min64) return 64;
    if (min32 >= 0 && tiles(32) >= min32) return 32;
    return GFX803_SGEMM_TILE;
}

// One wavefront per tile: gfx803_w_kernel computes a 64 x (8*PN) tile of C
// with 64 threads, each owning an 8 x PN block (PN = 8 or 4), and takes over
// the calls gfx803_w_applies picks from the 16x16-thread 64x64 tiles.
//
// Why a different tile instead of tuning the 4x4-per-thread ones:
//   - LDS bandwidth. A 4x4 block reads two float4 from LDS per 16
//     multiply-adds. A wave64 ds_read_b128 returns 1 KiB and the LDS returns
//     at most 128 bytes per clock to the whole CU, so at the full multiply-add
//     rate of four SIMDs the 4x4 tiles need all of it, and every LDS store or
//     conflict comes straight off the multiply-add rate. An 8x8 block reads
//     four float4 per 64 multiply-adds (half the LDS bandwidth), 8x4 three
//     per 32 (three quarters).
//   - Barriers. gfx8 cannot back off a barrier, so the compiler drains every
//     outstanding load (s_waitcnt vmcnt(0) lgkmcnt(0)) before each s_barrier,
//     and the 16x16-thread tiles hit two of those per K slice. A workgroup of
//     one wavefront needs none: LDS executes one wavefront's instructions in
//     issue order, so a store issued after the reads of a buffer cannot
//     overtake them, and reads issued after the store see it. gfx803_w_order
//     only keeps the compiler from moving LDS accesses across that point; it
//     emits no instruction.
//   - Latency hiding inside the wavefront. LDS holds two 8-deep K pieces; the
//     operands of the piece two ahead are loaded into registers in their
//     storage type (fp16 stays fp16) while the current piece is multiplied,
//     and widened only at the LDS store that follows that piece, so each
//     global load has a whole piece (512 multiply-adds per lane) to arrive.
//     The next K step's LDS reads are issued before the current step's 64 (or
//     32) multiply-adds, which cover the LDS latency.
// A tile on one wavefront lives on one SIMD, so a GEMM with few tiles
// balances worse over the 4 * CUs SIMDs than the 16x16-thread tiles, whose
// tile spreads over a CU's four SIMDs; gfx803_w_pick_pn takes the 64x32 tile
// when that halves the imbalance.
//
// Every output still accumulates its K products one at a time in increasing
// k in fp32 (v_mac_f32: product and sum each rounded, as in the other tiles),
// then alpha * acc + beta * C rounded once. The K tail is padded with zero
// products to a multiple of 8; the older tiles pad to 16 or 32, and a padded
// +0 turns an accumulated -0 into +0, so pz adds one +0 where the tile this
// replaces would have padded and this one does not. Results are then
// bit-identical to the older tiles.
#define GFX803_W_KH 8

__device__ __forceinline__ void gfx803_w_order() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

__device__ const float gfx803_w_zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};

// One operand's 8 x R piece (R = 64 rows of op(A), or 8*PN columns of
// op(B)), fetched by the 64 lanes in 16-byte (or 8-byte) vectors along the
// operand's contiguous index and stored to LDS as Xs[k][r]. KC: K is the
// contiguous index.
template <typename E, bool KC, int R>
struct gfx803_w_op {
    static constexpr int KH = GFX803_W_KH;
    static constexpr int PER = R * KH / 64;
    static constexpr int VW = PER < 16 / (int)sizeof(E) ? PER : 16 / (int)sizeof(E);
    static constexpr int NV = PER / VW;
    // Lane l's vector i sits at (r0 + DR*i, k0 + DK*i): the deltas are the
    // same for every lane, so the other vectors' LDS stores are immediate
    // offsets and their global addresses one uniform offset each. Registers
    // per vector push the fp32 8x8 tile past 128 VGPRs, and the spill reload
    // that follows waits (vmcnt counts in order) for the operand loads just
    // issued.
    static constexpr int KV = KC ? KH / VW : 1, RV = KC ? 1 : R / VW;
    static constexpr int DR = KC ? 64 / KV : 0, DK = KC ? 0 : 64 / RV;
    static_assert(PER % VW == 0 && (KC || VW % 4 == 0) && 64 % KV == 0 && 64 % RV == 0,
                  "piece must split into whole vectors");
    typedef E vec __attribute__((ext_vector_type(VW)));
    struct raw { vec v[NV]; };
    int r0, k0;
    const E* p;
    size_t off;
    size_t step;

    __device__ __forceinline__ explicit gfx803_w_op(int lane) {
        if constexpr (KC) { r0 = lane / KV; k0 = lane % KV * VW; }
        else { r0 = lane % RV * VW; k0 = lane / RV; }
    }

    __device__ __forceinline__ void init(const E* X, int ld, int row0) {
        p = KC ? X + (size_t)(row0 + r0) * ld + k0 : X + (size_t)k0 * ld + (row0 + r0);
        off = KC ? (size_t)DR * ld : (size_t)DK * ld;
        step = KC ? (size_t)KH : (size_t)KH * ld;
    }

    // The tile and the piece lie inside the matrix: plain vector loads. The
    // vectors need no alignment (see gfx803_ld).
    __device__ __forceinline__ raw load() {
        raw o;
        #pragma unroll
        for (int i = 0; i < NV; i++) o.v[i] = *reinterpret_cast<const vec*>(p + i * off);
        p += step;
        return o;
    }

    __device__ __forceinline__ void store(float (*Xs)[R], const raw& o) const {
        #pragma unroll
        for (int i = 0; i < NV; i++) {
            if constexpr (KC) {
                #pragma unroll
                for (int e = 0; e < VW; e++) Xs[k0 + e][r0 + DR * i] = static_cast<float>(o.v[i][e]);
            } else {
                #pragma unroll
                for (int q = 0; q < VW / 4; q++)
                    *reinterpret_cast<float4*>(&Xs[k0 + DK * i][r0 + 4 * q]) =
                        make_float4(static_cast<float>(o.v[i][4 * q + 0]), static_cast<float>(o.v[i][4 * q + 1]),
                                    static_cast<float>(o.v[i][4 * q + 2]), static_cast<float>(o.v[i][4 * q + 3]));
            }
        }
    }

    // Edge tiles and the K tail, one element at a time. An element outside
    // the matrix is read from gfx803_w_zero rather than selected to zero
    // after the load: the compiler turns a select whose operand is a load
    // into a per-lane branch around the load and waits for the load inside
    // it, which serializes the fetch. The values stay unpacked so nothing
    // touches them before the LDS store.
    struct graw { E v[NV * VW]; };
    __device__ __forceinline__ graw load_guarded(const E* X, int ld, int rows, int K, int row0, int kb) const {
        graw o;
        const E* zero = reinterpret_cast<const E*>(gfx803_w_zero);
        #pragma unroll
        for (int i = 0; i < NV; i++)
            #pragma unroll
            for (int e = 0; e < VW; e++) {
                const int rr = row0 + r0 + DR * i + (KC ? 0 : e), kk = kb + k0 + DK * i + (KC ? e : 0);
                const bool ok = rr < rows && kk < K;
                const size_t idx = ok ? (KC ? (size_t)rr * ld + kk : (size_t)kk * ld + rr) : 0;
                o.v[i * VW + e] = *(ok ? X + idx : zero);
            }
        return o;
    }

    __device__ __forceinline__ void store(float (*Xs)[R], const graw& o) const {
        #pragma unroll
        for (int i = 0; i < NV; i++) {
            if constexpr (KC) {
                #pragma unroll
                for (int e = 0; e < VW; e++) Xs[k0 + e][r0 + DR * i] = static_cast<float>(o.v[i * VW + e]);
            } else {
                #pragma unroll
                for (int q = 0; q < VW / 4; q++)
                    *reinterpret_cast<float4*>(&Xs[k0 + DK * i][r0 + 4 * q]) =
                        make_float4(static_cast<float>(o.v[i * VW + 4 * q + 0]),
                                    static_cast<float>(o.v[i * VW + 4 * q + 1]),
                                    static_cast<float>(o.v[i * VW + 4 * q + 2]),
                                    static_cast<float>(o.v[i * VW + 4 * q + 3]));
            }
        }
    }
};

// Lane (tx, ty) = (lane & 7, lane >> 3) owns rows tx*4 + {0..3} and
// 32 + tx*4 + {0..3}, and columns ty*4 + {0..3} (and 32 + ty*4 + {0..3} for
// PN = 8): every float4 LDS read covers 128 contiguous bytes across the 8
// distinct addresses of a wavefront, one pass over the 32 banks.
// G: guarded loads and stores (edge tile or K not a multiple of 8).
// P2 (unguarded tiles, K % 16 == 0): each K-contiguous operand is fetched two
// pieces at a time, the two loads back to back. One 8-deep piece of a
// K-contiguous fp16 row is 16 bytes of a 64-byte line, and with eight
// wavefronts per CU streaming 128 such rows each, the 16 KiB L1 drops the
// line before the next piece reads it, so L2 delivers each line up to four
// times; back-to-back loads use it twice per delivery. Costs four VGPRs per
// such operand.
template <typename E, bool TransA, bool TransB, int PN, bool G, bool P2 = false>
__device__ __forceinline__ void gfx803_w_tile(int M, int N, int K, float alpha,
                                              const E* __restrict__ A, int lda,
                                              const E* __restrict__ B, int ldb,
                                              float beta, E* __restrict__ C, int ldc,
                                              int row0, int col0, int pz,
                                              float (*As)[GFX803_W_KH][64],
                                              float (*Bs)[GFX803_W_KH][8 * PN]) {
    constexpr int TN = 8 * PN, KH = GFX803_W_KH;
    typedef gfx803_w_op<E, TransA, 64> OA;
    typedef gfx803_w_op<E, !TransB, TN> OB;
    const int lane = threadIdx.x, tx = lane & 7, ty = lane >> 3;
    OA oa(lane);
    OB ob(lane);
    if constexpr (!G) {
        oa.init(A, lda, row0);
        ob.init(B, ldb, col0);
    }
    auto load_a = [&](int h) {
        if constexpr (G) return oa.load_guarded(A, lda, M, K, row0, KH * h);
        else return oa.load();
    };
    auto load_b = [&](int h) {
        if constexpr (G) return ob.load_guarded(B, ldb, N, K, col0, KH * h);
        else return ob.load();
    };

    float acc[8][PN];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < PN; j++) acc[i][j] = 0.0f;

    auto compute = [&](const float (*as)[64], const float (*bs)[TN]) {
        float a[2][8], b[2][PN];
        auto read = [&](int s, int kk) {
#if GFX803_W_ABL_NO_LDS_READ
            const int k = 0 * kk;
#else
            const int k = kk;
#endif
            const float4 x0 = *reinterpret_cast<const float4*>(&as[k][tx * 4]);
            const float4 x1 = *reinterpret_cast<const float4*>(&as[k][32 + tx * 4]);
            a[s][0] = x0.x; a[s][1] = x0.y; a[s][2] = x0.z; a[s][3] = x0.w;
            a[s][4] = x1.x; a[s][5] = x1.y; a[s][6] = x1.z; a[s][7] = x1.w;
            const float4 y0 = *reinterpret_cast<const float4*>(&bs[k][ty * 4]);
            b[s][0] = y0.x; b[s][1] = y0.y; b[s][2] = y0.z; b[s][3] = y0.w;
            if constexpr (PN == 8) {
                const float4 y1 = *reinterpret_cast<const float4*>(&bs[k][32 + ty * 4]);
                b[s][4] = y1.x; b[s][5] = y1.y; b[s][6] = y1.z; b[s][7] = y1.w;
            }
        };
        read(0, 0);
        #pragma unroll
        for (int k = 0; k < KH; k++) {
            if (k + 1 < KH) read((k + 1) & 1, k + 1);
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < PN; j++) acc[i][j] += a[k & 1][i] * b[k & 1][j];
            // Left free, the scheduler issues all eight steps' LDS reads first
            // and groups the multiply-adds by accumulator, which keeps 128
            // operand values live next to the 64 sums and spills.
#if !GFX803_W_ABL_NO_LDS_READ
            __builtin_amdgcn_sched_barrier(0);
#endif
        }
    };

    const int nh = (K + KH - 1) / KH;
    {
        auto ra0 = load_a(0);
        auto rb0 = load_b(0);
        decltype(ra0) ra1;
        decltype(rb0) rb1;
        if (nh > 1) {
            ra1 = load_a(1);
            rb1 = load_b(1);
        }
        oa.store(As[0], ra0);
        ob.store(Bs[0], rb0);
        if (nh > 1) {
            oa.store(As[1], ra1);
            ob.store(Bs[1], rb1);
        }
    }
    if constexpr (P2 && !G) {
        constexpr bool PA = TransA, PB = !TransB;
        typename OA::raw a_lo, a_hi;
        typename OB::raw b_lo, b_hi;
        auto store_lo = [&](int buf) {
            oa.store(As[buf], a_lo);
            ob.store(Bs[buf], b_lo);
        };
        auto store_hi = [&](int buf) {
            oa.store(As[buf], PA ? a_hi : a_lo);
            ob.store(Bs[buf], PB ? b_hi : b_lo);
        };
        // Pieces 0 and 1 were stored above; nh is even.
        if (nh > 2) {
            a_lo = oa.load();
            if constexpr (PA) a_hi = oa.load();
            b_lo = ob.load();
            if constexpr (PB) b_hi = ob.load();
        }
        int t = 0;
        for (; t + 5 < nh; t += 2) {
            gfx803_w_order();
            compute(As[0], Bs[0]);
            gfx803_w_order();
            store_lo(0);
            if constexpr (!PA) a_lo = oa.load();
            if constexpr (!PB) b_lo = ob.load();
            gfx803_w_order();
            compute(As[1], Bs[1]);
            gfx803_w_order();
            store_hi(1);
            a_lo = oa.load();
            if constexpr (PA) a_hi = oa.load();
            b_lo = ob.load();
            if constexpr (PB) b_hi = ob.load();
        }
        if (t + 2 < nh) {
            gfx803_w_order();
            compute(As[0], Bs[0]);
            gfx803_w_order();
            store_lo(0);
            if constexpr (!PA) a_lo = oa.load();
            if constexpr (!PB) b_lo = ob.load();
            gfx803_w_order();
            compute(As[1], Bs[1]);
            gfx803_w_order();
            store_hi(1);
            t += 2;
        }
        gfx803_w_order();
        compute(As[0], Bs[0]);
        gfx803_w_order();
        compute(As[1], Bs[1]);
    } else {
        std::conditional_t<G, typename OA::graw, typename OA::raw> ra;
        std::conditional_t<G, typename OB::graw, typename OB::raw> rb;
        if (nh > 2) {
            ra = load_a(2);
            rb = load_b(2);
        }
        // The steady-state body is one basic block. With the LDS store and the
        // next fetch under a branch, the compiler sinks the multiply-adds below
        // that branch, which keeps all of a piece's LDS operands live at once
        // and spills.
        int t = 0;
        for (; t + 3 < nh; t++) {
            const int buf = t & 1;
            gfx803_w_order();
            compute(As[buf], Bs[buf]);
            gfx803_w_order();
#if !GFX803_W_ABL_NO_STORE
            oa.store(As[buf], ra);
            ob.store(Bs[buf], rb);
#endif
#if !GFX803_W_ABL_NO_FETCH
            ra = load_a(t + 3);
            rb = load_b(t + 3);
#endif
        }
        if (t + 2 < nh) {
            gfx803_w_order();
            compute(As[t & 1], Bs[t & 1]);
            gfx803_w_order();
            oa.store(As[t & 1], ra);
            ob.store(Bs[t & 1], rb);
            t++;
        }
        for (; t < nh; t++) {
            gfx803_w_order();
            compute(As[t & 1], Bs[t & 1]);
        }
    }

    if (pz) {
        #pragma unroll
        for (int i = 0; i < 8; i++)
            #pragma unroll
            for (int j = 0; j < PN; j++) acc[i][j] += 0.0f;
    }

    #pragma unroll
    for (int j = 0; j < PN; j++) {
        const int col = col0 + (j < 4 ? ty * 4 + j : 32 + ty * 4 + j - 4);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int row = row0 + (i < 4 ? tx * 4 + i : 32 + tx * 4 + i - 4);
            if (!G || (row < M && col < N)) {
                const size_t idx = (size_t)col * ldc + row;
                C[idx] = static_cast<E>(alpha * acc[i][j] + (beta != 0.0f ? beta * static_cast<float>(C[idx]) : 0.0f));
            }
        }
    }
}

// Edge = false: every tile of the launch is interior and K % 8 == 0, so no
// tile runs the guarded code. P2 requires K % 16 == 0. Tile order as in
// gfx803_sgemm_tile_g (GW column tiles walk the row tiles together).
template <typename E, bool TransA, bool TransB, int PN, bool Edge, bool P2>
__global__ void __launch_bounds__(64) __attribute__((amdgpu_waves_per_eu(2)))
gfx803_w_kernel(int GW, int M, int N, int K, float alpha,
                const E* __restrict__ A, int lda, int64_t strideA,
                const E* __restrict__ B, int ldb, int64_t strideB,
                float beta, E* __restrict__ C, int ldc, int64_t strideC, int pz) {
    constexpr int TM = 64, TN = 8 * PN;
    __shared__ float As[2][GFX803_W_KH][TM];
    __shared__ float Bs[2][GFX803_W_KH][TN];
    const int64_t b = (int64_t)blockIdx.z;
    A += b * strideA;
    B += b * strideB;
    C += b * strideC;
    const int gx = (M + TM - 1) / TM, gy = (N + TN - 1) / TN;
    const int per = GW * gx, grp = blockIdx.x / per, first = grp * GW;
    const int gw = gy - first < GW ? gy - first : GW, loc = blockIdx.x - grp * per;
    const int row0 = (loc / gw) * TM, col0 = (first + loc % gw) * TN;
    if (!Edge || (row0 + TM <= M && col0 + TN <= N && K % GFX803_W_KH == 0))
        gfx803_w_tile<E, TransA, TransB, PN, false, P2>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
                                                        row0, col0, pz, As, Bs);
    else
        gfx803_w_tile<E, TransA, TransB, PN, true>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
                                                   row0, col0, pz, As, Bs);
}

// Two wavefronts per tile: gfx803_w2_kernel computes a 64x128 tile of an fp16
// TN product (both operands contiguous in K), each wavefront a 64x64 half with
// the 8x8-per-thread layout of gfx803_w_tile, the two sharing the A piece.
//
// Why: gfx803_w_kernel fetches 8-deep K pieces, 16 bytes of each 64-byte
// line of a K-contiguous fp16 row, and L2 delivers every line up to four
// times; fetching two pieces at a time (P2) takes it to 127 VGPRs, so a
// wider fetch does not fit one wavefront. Here a slice is 16 deep (32 bytes
// per row), and the 128 threads share its fetch: 12 VGPRs hold the next
// slice while the current one is multiplied. One LDS buffer (12 KiB) leaves
// room for four workgroups per CU, two wavefronts per SIMD. The barriers
// drain the outstanding loads, which by then have had a whole slice (1024
// multiply-adds per lane) to arrive.
//
// Each output accumulates its K products one at a time in increasing k, as in
// the other tiles, and pz adds the +0 those would have padded, so results are
// bit-identical.
#define GFX803_W2_KH 16

template <typename E>
__global__ void __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(2)))
gfx803_w2_kernel(int M, int N, int K, float alpha,
                 const E* __restrict__ A, int lda, int64_t strideA,
                 const E* __restrict__ B, int ldb, int64_t strideB,
                 float beta, E* __restrict__ C, int ldc, int64_t strideC, int pz) {
    static_assert(sizeof(E) == 2, "fp16 operands");
    constexpr int KH = GFX803_W2_KH;
    typedef E vec __attribute__((ext_vector_type(8)));
    __shared__ float As[KH][64];
    __shared__ float Bs[KH][128];
    const int64_t bz = (int64_t)blockIdx.z;
    A += bz * strideA;
    B += bz * strideB;
    C += bz * strideC;
    const int gx = M / 64;
    const int row0 = (blockIdx.x % gx) * 64, col0 = (blockIdx.x / gx) * 128;
    const int t = threadIdx.x, r = t >> 1, kq = (t & 1) * 8;
    const E* pa = A + (size_t)(row0 + r) * lda + kq;
    const E* pb0 = B + (size_t)(col0 + r) * ldb + kq;
    const E* pb1 = B + (size_t)(col0 + 64 + r) * ldb + kq;
    const int w = t >> 6, lane = t & 63, tx = lane & 7, ty = lane >> 3, cb = 64 * w;

    vec ra, rb0, rb1;
    auto load = [&]() {
        ra = *reinterpret_cast<const vec*>(pa);
        rb0 = *reinterpret_cast<const vec*>(pb0);
        rb1 = *reinterpret_cast<const vec*>(pb1);
        pa += KH;
        pb0 += KH;
        pb1 += KH;
    };
    auto store = [&]() {
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            As[kq + e][r] = static_cast<float>(ra[e]);
            Bs[kq + e][r] = static_cast<float>(rb0[e]);
            Bs[kq + e][64 + r] = static_cast<float>(rb1[e]);
        }
    };

    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

    auto compute = [&]() {
        float a[2][8], b[2][8];
        auto read = [&](int s, int k) {
            const float4 x0 = *reinterpret_cast<const float4*>(&As[k][tx * 4]);
            const float4 x1 = *reinterpret_cast<const float4*>(&As[k][32 + tx * 4]);
            const float4 y0 = *reinterpret_cast<const float4*>(&Bs[k][cb + ty * 4]);
            const float4 y1 = *reinterpret_cast<const float4*>(&Bs[k][cb + 32 + ty * 4]);
            a[s][0] = x0.x; a[s][1] = x0.y; a[s][2] = x0.z; a[s][3] = x0.w;
            a[s][4] = x1.x; a[s][5] = x1.y; a[s][6] = x1.z; a[s][7] = x1.w;
            b[s][0] = y0.x; b[s][1] = y0.y; b[s][2] = y0.z; b[s][3] = y0.w;
            b[s][4] = y1.x; b[s][5] = y1.y; b[s][6] = y1.z; b[s][7] = y1.w;
        };
        read(0, 0);
        #pragma unroll
        for (int k = 0; k < KH; k++) {
            if (k + 1 < KH) read((k + 1) & 1, k + 1);
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] += a[k & 1][i] * b[k & 1][j];
            __builtin_amdgcn_sched_barrier(0);
        }
    };

    const int ns = K / KH;
    load();
    store();
    __syncthreads();
    // One basic block per slice (see gfx803_w_tile): the last slice is
    // computed after the loop instead of under a branch.
    for (int s = 0; s + 1 < ns; s++) {
        load();
        compute();
        __syncthreads();
        store();
        __syncthreads();
    }
    compute();

    if (pz) {
        #pragma unroll
        for (int i = 0; i < 8; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++) acc[i][j] += 0.0f;
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int col = col0 + cb + (j < 4 ? ty * 4 + j : 32 + ty * 4 + j - 4);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int row = row0 + (i < 4 ? tx * 4 + i : 32 + tx * 4 + i - 4);
            const size_t idx = (size_t)col * ldc + row;
            C[idx] = static_cast<E>(alpha * acc[i][j] + (beta != 0.0f ? beta * static_cast<float>(C[idx]) : 0.0f));
        }
    }
}

inline long gfx803_sgemm_cus() {
    static const long cus = [] {
        int dev = 0, n = 0;
        if (hipGetDevice(&dev) != hipSuccess ||
            hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess ||
            n <= 0)
            n = 36;
        return (long)n;
    }();
    return cus;
}

// The calls gfx803_w2_kernel takes: fp16 TN with whole 64x128 tiles, K a
// multiple of 16, and at least three rounds of tiles over the CUs' four
// workgroup slots each. On an RX 590 it ran SDXL's large linears 17-27%
// faster (TN 5120x8192x640: 13.9 -> 10.1 ms, 10240x2048x1280: 10.4 -> 8.6)
// and calls with fewer tiles up to 64% slower: a 64x128 tile is twice the
// work of the 64x64 tiles it replaces, so a call with few of them leaves
// CUs idle in its last round. GFX803_SGEMM_W2 overrides the rule for timing:
// 0 = never, 1 = every call whose shape it supports.
inline bool gfx803_w2_applies(int M, int N, int K, int batch, bool transA, bool transB, size_t elem) {
    static const long mode = gfx803_sgemm_env_long("GFX803_SGEMM_W2", -1);
    if (mode == 0 || elem != 2 || !transA || transB || M % 64 != 0 || N % 128 != 0 || K % GFX803_W2_KH != 0)
        return false;
    return mode == 1 || (long)(M / 64) * (N / 128) * batch >= 3 * 4 * gfx803_sgemm_cus();
}

// Tile width for gfx803_w_kernel: 64x64 tiles unless 64x32 tiles fill the
// 4 * CUs SIMDs in markedly fewer rounds. A tile runs on one SIMD, so the
// GEMM takes ceil(tiles / SIMDs) tile times; a 64x32 tile takes about half a
// 64x64 tile's time, divided by its lower efficiency (three LDS reads per 32
// multiply-adds against four per 64). An fp32 launch with edge tiles takes the
// 64x32 tile: the fp32 64x64 edge kernel needs more than 128 VGPRs and
// reloads a spilled value inside its loop, and since vector memory counts
// complete in order, waiting for that reload also waits for the operand loads
// just issued. GFX803_SGEMM_WAVE_PN=4 or 8 forces one.
inline int gfx803_w_pick_pn(int M, int N, int K, int batch, size_t elem) {
    static const long forced = gfx803_sgemm_env_long("GFX803_SGEMM_WAVE_PN", 0);
    if (forced == 4 || forced == 8) return (int)forced;
    if (elem == 4 && (M % 64 != 0 || N % 64 != 0 || K % GFX803_W_KH != 0)) return 4;
    const double simds = 4.0 * gfx803_sgemm_cus();
    const double gx = (M + 63) / 64;
    const double t8 = gx * ((N + 63) / 64) * batch, t4 = gx * ((N + 31) / 32) * batch;
    const double eff4 = 0.85;
    return __builtin_ceil(t4 / simds) * 0.5 / eff4 < __builtin_ceil(t8 / simds) ? 4 : 8;
}

// Which calls gfx803_w_kernel takes from the 16x16-thread tiles. Without
// barriers or a second wavefront in its workgroup, nothing overlaps a tile's
// global loads with its multiply-adds except other tiles on the same SIMD, so
// it wins only where a call has many tiles per SIMD. On an RX 590, fp16
// strided-batched NN (MIOpen's conv-as-GEMM) with batch >= 8 and M >= 512 ran
// up to 28% faster (SDXL 8192x320x640 b16: 13.6 -> 9.8 ms), a few such shapes
// up to 18% slower, and the SDXL and SD1.5 UNet call logs' GEMM time dropped
// 2%. Smaller or less batched shapes lose. On single GEMMs it lost up
// to 3.3x (TN 1280x2048x5120: 6.6 -> 21.5 ms); on TN 1280x2048x1280 removing
// its global fetch took it from 3.3 to 5.6 TFLOPS and removing its LDS reads
// changed nothing. In fp32 batched NN it ran 0.5-2.8x the old time with no
// pattern a rule could follow. GFX803_SGEMM_WAVE overrides the rule for timing: 0 = never, 1 =
// every call given the 64x64 tile, 3 = also those given the 32x32 tile.
inline bool gfx803_w_applies(int tile, int M, bool transA, bool transB, int batch, size_t elem) {
    static const long mode = gfx803_sgemm_env_long("GFX803_SGEMM_WAVE", -1);
    if (mode >= 0) return (mode == 1 && tile == 64) || (mode == 3 && tile >= 32);
    return tile == 64 && elem == 2 && !transA && !transB && batch >= 8 && M >= 512;
}

// Runs a whole (strided-batched) GEMM on gfx803_w_kernel. kt_old is the K
// slice depth of the tile the call would otherwise run on (16 or 32), for pz.
// Returns false, launching nothing, when the grid would not fit.
template <typename E>
inline bool gfx803_w_gemm(int M, int N, int K, float alpha,
                          const E* A, int lda, int64_t strideA, bool transA,
                          const E* B, int ldb, int64_t strideB, bool transB,
                          float beta, E* C, int ldc, int64_t strideC,
                          int batch_count, int kt_old, hipStream_t stream, hipError_t* err) {
    const int PN = gfx803_w_pick_pn(M, N, K, batch_count, sizeof(E)), TN = 8 * PN;
    const long gx = (M + 63) / 64;
    const long max_tiles = (0xffffffffL / 64) & ~0xffL;
    if (gx > max_tiles) return false;
    long gy_max = max_tiles / gx;
    if (gy_max > (long)(0x7fffffff / TN)) gy_max = 0x7fffffff / TN;
    const int max_cols = (int)(gy_max * TN);
    const int pz = K % GFX803_W_KH == 0 && K % kt_old != 0;
    const bool a_big = (size_t)M * K * sizeof(E) > (2u << 20);
    const bool b_rows_aligned = transB || ((size_t)ldb * sizeof(E)) % 256 == 0;
    static const long gw_env = gfx803_sgemm_env_long("GFX803_SGEMM_WAVE_GW", 0);
    static const long kc2_env = gfx803_sgemm_env_long("GFX803_SGEMM_WAVE_KC2", 0);
    const bool p2 = kc2_env != 0 && sizeof(E) == 2 && K % 16 == 0 && (transA || !transB);
    const int GW = gw_env > 0 ? (int)gw_env : (transA && a_big && b_rows_aligned ? 8 : 1);
    (void)hipGetLastError();
    for (int c0 = 0; c0 < N; c0 += max_cols) {
        const int cols = N - c0 < max_cols ? N - c0 : max_cols;
        const E* Bc = B + (transB ? (int64_t)c0 : (int64_t)c0 * ldb);
        E* Cc = C + (int64_t)c0 * ldc;
        const unsigned tiles = (unsigned)(gx * ((cols + TN - 1) / TN));
        const bool edge = M % 64 != 0 || cols % TN != 0 || K % GFX803_W_KH != 0;
        for (int start = 0; start < batch_count; start += GFX803_SGEMM_MAX_GRID_YZ) {
            int chunk = batch_count - start;
            if (chunk > GFX803_SGEMM_MAX_GRID_YZ) chunk = GFX803_SGEMM_MAX_GRID_YZ;
            const dim3 grid(tiles, 1, (unsigned)chunk);
            const E* As = A + (int64_t)start * strideA;
            const E* Bs = Bc + (int64_t)start * strideB;
            E* Cs = Cc + (int64_t)start * strideC;
#define GFX803_W(TA, TB, P, ED, P2) \
    hipLaunchKernelGGL((gfx803_w_kernel<E, TA, TB, P, ED, P2>), grid, dim3(64), 0, stream, GW, M, cols, K, alpha, \
                       As, lda, strideA, Bs, ldb, strideB, beta, Cs, ldc, strideC, pz)
#define GFX803_W_2(TA, TB, P, ED) do { \
        if (sizeof(E) == 2 && (TA || !TB) && p2) GFX803_W(TA, TB, P, ED, (sizeof(E) == 2 && (TA || !TB))); \
        else GFX803_W(TA, TB, P, ED, false); \
    } while (0)
#define GFX803_W_E(TA, TB, P) do { if (edge) GFX803_W_2(TA, TB, P, true); else GFX803_W_2(TA, TB, P, false); } while (0)
#define GFX803_W_P(TA, TB) do { if (PN == 8) GFX803_W_E(TA, TB, 8); else GFX803_W_E(TA, TB, 4); } while (0)
            if (transA) {
                if (transB) GFX803_W_P(true, true); else GFX803_W_P(true, false);
            } else {
                if (transB) GFX803_W_P(false, true); else GFX803_W_P(false, false);
            }
#undef GFX803_W_P
#undef GFX803_W_E
#undef GFX803_W_2
#undef GFX803_W
            *err = hipGetLastError();
            if (*err != hipSuccess) return true;
        }
    }
    *err = hipSuccess;
    return true;
}
// C = beta * C, without reading A or B: what BLAS defines for alpha == 0 or
// K == 0. Running the GEMM kernels instead would compute 0 * (A * B), which
// turns an Inf or NaN anywhere in A or B into NaN in C. beta == 0 writes zeros
// without reading C, as the GEMM kernels do.
template <typename T>
__global__ void gfx803_scale_c_kernel(int M, int N, int batch, float beta,
                                      T* __restrict__ C, int ldc, int64_t strideC) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    for (int b = blockIdx.z; b < batch; b += gridDim.z) {
        for (int j = blockIdx.y; j < N; j += gridDim.y) {
            T* c = C + (int64_t)b * strideC + (int64_t)j * ldc + i;
            *c = static_cast<T>(beta != 0.0f ? beta * static_cast<float>(*c) : 0.0f);
        }
    }
}

template <typename T>
inline hipError_t gfx803_scale_c(int M, int N, float beta, T* C, int ldc, int64_t strideC,
                                 int batch_count, hipStream_t stream) {
    if (beta == 1.0f) return hipSuccess;
    (void)hipGetLastError();
    const unsigned gy = (unsigned)(N < GFX803_SGEMM_MAX_GRID_YZ ? N : GFX803_SGEMM_MAX_GRID_YZ);
    const unsigned gz = (unsigned)(batch_count < GFX803_SGEMM_MAX_GRID_YZ ? batch_count : GFX803_SGEMM_MAX_GRID_YZ);
    hipLaunchKernelGGL(gfx803_scale_c_kernel<T>, dim3((unsigned)((M + 255) / 256), gy, gz), dim3(256), 0, stream,
                       M, N, batch_count, beta, C, ldc, strideC);
    return hipGetLastError();
}

// Batch is split into launches of at most GFX803_SGEMM_MAX_GRID_YZ matrices
// (grid Z limit), each launch's base pointers advanced by chunk_start*stride,
// and C's columns as described at gfx803_sgemm.
// T is the storage type of A, B and C: float, or _Float16 for the fp16
// route. Products and sums are fp32 either way and C is rounded once, when it
// is stored, which is what rocBLAS does for fp16 with fp32 compute.
template <typename T>
inline hipError_t gfx803_gemm_strided_batched(int M, int N, int K,
                                              float alpha,
                                              const T* A, int lda, int64_t strideA, bool transA,
                                              const T* B, int ldb, int64_t strideB, bool transB,
                                              float beta,
                                              T* C, int ldc, int64_t strideC,
                                              int batch_count,
                                              hipStream_t stream = 0) {
    if (alpha == 0.0f || K == 0) return gfx803_scale_c<T>(M, N, beta, C, ldc, strideC, batch_count, stream);
    if (gfx803_small_n_applies(N, transA)) {
        (void)hipGetLastError();
        for (int start = 0; start < batch_count; start += GFX803_SGEMM_MAX_GRID_YZ) {
            int chunk = batch_count - start;
            if (chunk > GFX803_SGEMM_MAX_GRID_YZ) chunk = GFX803_SGEMM_MAX_GRID_YZ;
            gfx803_small_n_dispatch<T>(N, (unsigned)chunk, stream, M, K, alpha,
                                       A + (int64_t)start * strideA, lda, strideA, transB,
                                       B + (int64_t)start * strideB, ldb, strideB, beta,
                                       C + (int64_t)start * strideC, ldc, strideC);
            hipError_t err = hipGetLastError();
            if (err != hipSuccess) return err;
        }
        return hipSuccess;
    }
    const int tile = gfx803_sgemm_pick_tile(M, N, K, batch_count, sizeof(T) == 2);
    if constexpr (sizeof(T) == 2) {
        if (gfx803_w2_applies(M, N, K, batch_count, transA, transB, sizeof(T)) &&
            (int64_t)(M / 64) * (N / 128) <= 0x7fffffff) {
            const int kt_old = gfx803_sgemm_g_applies<T>(tile, M, K, lda, transA, ldb, transB) ? 32 : GFX803_SGEMM_BIG_K;
            (void)hipGetLastError();
            for (int start = 0; start < batch_count; start += GFX803_SGEMM_MAX_GRID_YZ) {
                int chunk = batch_count - start;
                if (chunk > GFX803_SGEMM_MAX_GRID_YZ) chunk = GFX803_SGEMM_MAX_GRID_YZ;
                hipLaunchKernelGGL(gfx803_w2_kernel<T>, dim3((unsigned)((M / 64) * (N / 128)), 1, (unsigned)chunk),
                                   dim3(128), 0, stream, M, N, K, alpha,
                                   A + (int64_t)start * strideA, lda, strideA,
                                   B + (int64_t)start * strideB, ldb, strideB, beta,
                                   C + (int64_t)start * strideC, ldc, strideC, K % kt_old != 0 ? 1 : 0);
                hipError_t err = hipGetLastError();
                if (err != hipSuccess) return err;
            }
            return hipSuccess;
        }
    }
    if (gfx803_w_applies(tile, M, transA, transB, batch_count, sizeof(T))) {
        const int kt_old = gfx803_sgemm_g_applies<T>(tile, M, K, lda, transA, ldb, transB) ? 32 : GFX803_SGEMM_BIG_K;
        hipError_t err;
        if (gfx803_w_gemm<T>(M, N, K, alpha, A, lda, strideA, transA, B, ldb, strideB, transB, beta, C, ldc,
                             strideC, batch_count, kt_old, stream, &err))
            return err;
    }
    const unsigned grid_x = (unsigned)((M + tile - 1) / tile);
    const int max_cols = GFX803_SGEMM_MAX_GRID_YZ * tile;

    (void)hipGetLastError();
    dim3 block(GFX803_SGEMM_TILE, GFX803_SGEMM_TILE);
    for (int c0 = 0; c0 < N; c0 += max_cols) {
        const int cols = N - c0 < max_cols ? N - c0 : max_cols;
        const T* Bc = B + (transB ? (int64_t)c0 : (int64_t)c0 * ldb);
        T* Cc = C + (int64_t)c0 * ldc;
        const unsigned grid_y = (unsigned)((cols + tile - 1) / tile);
        for (int start = 0; start < batch_count; start += GFX803_SGEMM_MAX_GRID_YZ) {
            int chunk = batch_count - start;
            if (chunk > GFX803_SGEMM_MAX_GRID_YZ) chunk = GFX803_SGEMM_MAX_GRID_YZ;
            dim3 grid(grid_x, grid_y, (unsigned)chunk);
            const T* As = A + (int64_t)start * strideA;
            const T* Bs = Bc + (int64_t)start * strideB;
            T* Cs = Cc + (int64_t)start * strideC;
            auto fast = [&](auto defer_tag) -> bool {
                constexpr bool D = decltype(defer_tag)::value;
                if (gfx803_sgemm_g_launch<T, D>(tile, grid, stream, M, cols, K, alpha, As, lda, strideA, transA,
                                                Bs, ldb, strideB, transB, beta, Cs, ldc, strideC))
                    return true;
                if (!transB && tile == 64) {
                    if (batch_count == 1) {
                        if (!transA) {
                            hipLaunchKernelGGL((gfx803_sgemm_p_fast_kernel<4, false, T, D>), grid, block, 0, stream,
                                               M, cols, K, alpha, As, lda, Bs, ldb, beta, Cs, ldc);
                        } else {
                            hipLaunchKernelGGL((gfx803_sgemm_p_fast_kernel<4, true, T, D>), grid, block, 0, stream,
                                               M, cols, K, alpha, As, lda, Bs, ldb, beta, Cs, ldc);
                        }
                    } else {
                        if (!transA) {
                            hipLaunchKernelGGL((gfx803_sgemm_p_strided_batched_fast_kernel<4, false, T, D>), grid, block, 0,
                                               stream, M, cols, K, alpha, As, lda, strideA, Bs, ldb, strideB, beta, Cs,
                                               ldc, strideC);
                        } else {
                            hipLaunchKernelGGL((gfx803_sgemm_p_strided_batched_fast_kernel<4, true, T, D>), grid, block, 0,
                                               stream, M, cols, K, alpha, As, lda, strideA, Bs, ldb, strideB, beta, Cs,
                                               ldc, strideC);
                        }
                    }
                    return true;
                }
                if (!transB && tile == 32) {
                    if (batch_count == 1) {
                        if (!transA) {
                            hipLaunchKernelGGL((gfx803_sgemm_p2_fast_kernel<false, T, D>), grid, block, 0, stream,
                                               M, cols, K, alpha, As, lda, Bs, ldb, beta, Cs, ldc);
                        } else {
                            hipLaunchKernelGGL((gfx803_sgemm_p2_fast_kernel<true, T, D>), grid, block, 0, stream,
                                               M, cols, K, alpha, As, lda, Bs, ldb, beta, Cs, ldc);
                        }
                    } else {
                        if (!transA) {
                            hipLaunchKernelGGL((gfx803_sgemm_p2_strided_batched_fast_kernel<false, T, D>), grid, block, 0,
                                               stream, M, cols, K, alpha, As, lda, strideA, Bs, ldb, strideB, beta, Cs,
                                               ldc, strideC);
                        } else {
                            hipLaunchKernelGGL((gfx803_sgemm_p2_strided_batched_fast_kernel<true, T, D>), grid, block, 0,
                                               stream, M, cols, K, alpha, As, lda, strideA, Bs, ldb, strideB, beta, Cs,
                                               ldc, strideC);
                        }
                    }
                    return true;
                }
                return false;
            };
            const bool launched = fast(std::bool_constant<sizeof(T) == 2>());
            if (!launched) {
                if (batch_count == 1) {
                    auto kernel = tile == 64 ? gfx803_sgemm_p_kernel<4, T>
                                : tile == 32 ? gfx803_sgemm_p_kernel<2, T>
                                             : gfx803_sgemm_kernel<T>;
                    hipLaunchKernelGGL(kernel, grid, block, 0, stream,
                                       M, cols, K, alpha, As, lda, transA, Bs, ldb, transB, beta, Cs, ldc);
                } else {
                    auto kernel = tile == 64 ? gfx803_sgemm_p_strided_batched_kernel<4, T>
                                : tile == 32 ? gfx803_sgemm_p_strided_batched_kernel<2, T>
                                             : gfx803_sgemm_strided_batched_kernel<T>;
                    hipLaunchKernelGGL(kernel, grid, block, 0, stream,
                                       M, cols, K, alpha, As, lda, strideA, transA, Bs, ldb, strideB, transB,
                                       beta, Cs, ldc, strideC);
                }
            }
            hipError_t err = hipGetLastError();
            if (err != hipSuccess) return err;
        }
    }
    return hipSuccess;
}

inline hipError_t gfx803_sgemm_strided_batched(int M, int N, int K,
                                               float alpha,
                                               const float* A, int lda, int64_t strideA, bool transA,
                                               const float* B, int ldb, int64_t strideB, bool transB,
                                               float beta,
                                               float* C, int ldc, int64_t strideC,
                                               int batch_count,
                                               hipStream_t stream = 0) {
    return gfx803_gemm_strided_batched<float>(M, N, K, alpha, A, lda, strideA, transA, B, ldb, strideB,
                                              transB, beta, C, ldc, strideC, batch_count, stream);
}

// C's columns go out in launches of at most GFX803_SGEMM_MAX_GRID_YZ tiles
// (grid Y limit), each with B and C advanced to its first column, so any N
// runs on these kernels. Handing a wide GEMM to rocBLAS instead would compute
// it on the Tensile kernels this shim exists to avoid; a 4x3 @ 3x4.9M filter
// transform is one such shape. Any error left pending by earlier unrelated
// work is cleared first, so a stale error can't be reported as a failure of
// this launch.
inline hipError_t gfx803_sgemm(int M, int N, int K,
                               float alpha,
                               const float* A, int lda, bool transA,
                               const float* B, int ldb, bool transB,
                               float beta,
                               float* C, int ldc,
                               hipStream_t stream = 0) {
    return gfx803_sgemm_strided_batched(M, N, K, alpha, A, lda, 0, transA, B, ldb, 0, transB,
                                        beta, C, ldc, 0, 1, stream);
}
