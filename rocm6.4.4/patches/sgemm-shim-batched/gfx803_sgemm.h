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
// and 32x32 tiles for fp32 with B not transposed, which is most of what
// PyTorch issues; see the comment at gfx803_sgemm_tile_p_fast.
// Every output accumulates the same products in the same order in all of
// them, so they return bit-identical results and gfx803_sgemm_pick_tile picks
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
// no runtime branch on layout. The float4 path needs no alignment: gfx8 runs
// in unaligned mode under amdkfd, and edge tiles fall back to guarded scalar
// loads.
template <int P, bool TransA, typename E>
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

    auto fetch = [&](int k0) {
        int gk_a = k0 + k_a;
        if constexpr (!TransA) {
            if (gr_a + 3 < M && gk_a < K) {
                pa_vec = *reinterpret_cast<const float4*>(&A[(size_t)gk_a * lda + gr_a]);
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
                pa_vec = *reinterpret_cast<const float4*>(&A[(size_t)gr_a * lda + gk_a]);
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
    fetch(0);
    for (int t = 0; t < n_tiles; t++) {
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

        __syncthreads();
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

// The 32x32 tile with the scheme of gfx803_sgemm_tile_p_fast. Each thread
// fetches two values of A and two of B per K slice, so the wide loads are
// float2.
template <bool TransA, typename E>
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

    auto fetch = [&](int k0) {
        int gk_a = k0 + k_a;
        if constexpr (!TransA) {
            if (gr_a + 1 < M && gk_a < K) {
                pa = *reinterpret_cast<const float2*>(&A[(size_t)gk_a * lda + gr_a]);
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
            pb = *reinterpret_cast<const float2*>(&B[(size_t)gc_b * ldb + gk_b]);
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
    fetch(0);
    for (int t = 0; t < n_tiles; t++) {
        *reinterpret_cast<float2*>(&As[k_a][r_a]) = pa;

        Bs[k_b + 0][c_b] = pb.x;
        Bs[k_b + 1][c_b] = pb.y;

        __syncthreads();
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

template <int P, bool TransA, typename T>
__global__ void __launch_bounds__(256)
gfx803_sgemm_p_fast_kernel(int M, int N, int K,
                           float alpha,
                           const T* __restrict__ A, int lda,
                           const T* __restrict__ B, int ldb,
                           float beta,
                           T* __restrict__ C, int ldc) {
    gfx803_sgemm_tile_p_fast<P, TransA, T>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <int P, bool TransA, typename T>
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
    gfx803_sgemm_tile_p_fast<P, TransA, T>(M, N, K, alpha,
                                          A + b * strideA, lda,
                                          B + b * strideB, ldb,
                                          beta,
                                          C + b * strideC, ldc);
}

template <bool TransA, typename T>
__global__ void __launch_bounds__(256)
gfx803_sgemm_p2_fast_kernel(int M, int N, int K,
                            float alpha,
                            const T* __restrict__ A, int lda,
                            const T* __restrict__ B, int ldb,
                            float beta,
                            T* __restrict__ C, int ldc) {
    gfx803_sgemm_tile_p2_fast<TransA, T>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <bool TransA, typename T>
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
    gfx803_sgemm_tile_p2_fast<TransA, T>(M, N, K, alpha,
                                         A + b * strideA, lda,
                                         B + b * strideB, ldb,
                                         beta,
                                         C + b * strideC, ldc);
}

static inline long gfx803_sgemm_env_long(const char* name, long fallback) {
    const char* v = getenv(name);
    return v ? atol(v) : fallback;
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
// GFX803_SGEMM_MIN_TILES64 and GFX803_SGEMM_MIN_TILES32 override the tile-
// count thresholds for benchmarking (0 = always, -1 = never use that tile).
inline int gfx803_sgemm_pick_tile(int M, int N, int K, int batch) {
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
    if (min64 >= 0 && ((K > 32 && M > 16 && N > 16) || min64_forced) && tiles(64) >= min64) return 64;
    if (min32 >= 0 && tiles(32) >= min32) return 32;
    return GFX803_SGEMM_TILE;
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
    const int tile = gfx803_sgemm_pick_tile(M, N, K, batch_count);
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
            if (!transB && tile == 64 && std::is_same_v<T, float>) {
                if (batch_count == 1) {
                    if (!transA) {
                        hipLaunchKernelGGL((gfx803_sgemm_p_fast_kernel<4, false, float>), grid, block, 0, stream,
                                           M, cols, K, alpha, (const float*)As, lda, (const float*)Bs, ldb, beta, (float*)Cs, ldc);
                    } else {
                        hipLaunchKernelGGL((gfx803_sgemm_p_fast_kernel<4, true, float>), grid, block, 0, stream,
                                           M, cols, K, alpha, (const float*)As, lda, (const float*)Bs, ldb, beta, (float*)Cs, ldc);
                    }
                } else {
                    if (!transA) {
                        hipLaunchKernelGGL((gfx803_sgemm_p_strided_batched_fast_kernel<4, false, float>), grid, block, 0, stream,
                                           M, cols, K, alpha, (const float*)As, lda, strideA, (const float*)Bs, ldb, strideB, beta, (float*)Cs, ldc, strideC);
                    } else {
                        hipLaunchKernelGGL((gfx803_sgemm_p_strided_batched_fast_kernel<4, true, float>), grid, block, 0, stream,
                                           M, cols, K, alpha, (const float*)As, lda, strideA, (const float*)Bs, ldb, strideB, beta, (float*)Cs, ldc, strideC);
                    }
                }
            } else if (!transB && tile == 32 && std::is_same_v<T, float>) {
                if (batch_count == 1) {
                    if (!transA) {
                        hipLaunchKernelGGL((gfx803_sgemm_p2_fast_kernel<false, float>), grid, block, 0, stream,
                                           M, cols, K, alpha, (const float*)As, lda, (const float*)Bs, ldb, beta, (float*)Cs, ldc);
                    } else {
                        hipLaunchKernelGGL((gfx803_sgemm_p2_fast_kernel<true, float>), grid, block, 0, stream,
                                           M, cols, K, alpha, (const float*)As, lda, (const float*)Bs, ldb, beta, (float*)Cs, ldc);
                    }
                } else {
                    if (!transA) {
                        hipLaunchKernelGGL((gfx803_sgemm_p2_strided_batched_fast_kernel<false, float>), grid, block, 0, stream,
                                           M, cols, K, alpha, (const float*)As, lda, strideA, (const float*)Bs, ldb, strideB, beta, (float*)Cs, ldc, strideC);
                    } else {
                        hipLaunchKernelGGL((gfx803_sgemm_p2_strided_batched_fast_kernel<true, float>), grid, block, 0, stream,
                                           M, cols, K, alpha, (const float*)As, lda, strideA, (const float*)Bs, ldb, strideB, beta, (float*)Cs, ldc, strideC);
                    }
                }
            } else {
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
// work is cleared first, so a stale error can't be mistaken for a failure of
// this launch (the shim would otherwise fall back to rocBLAS AFTER this
// kernel already ran, applying the GEMM twice when beta != 0).
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
