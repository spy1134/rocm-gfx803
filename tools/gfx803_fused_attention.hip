#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define WARP_SIZE 64

// Pins the issue order of a load group ahead of the FMAs that consume the
// previous group.  The builtin emits no instruction; the inline asm fallback
// is a plain compiler barrier with the same effect on the scheduler.
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_sched_barrier)
#define SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#else
#define SCHED_BARRIER() asm volatile("" ::: "memory")
#endif

// Lane exchange helpers.  xor masks below 32 use ds_swizzle, which needs no
// address register; the cross-half mask and the key-group broadcasts use
// ds_bpermute with the lane address in one VGPR and the compile-time part of
// the source lane folded into the instruction's immediate offset.
template <int MASK>
__device__ __forceinline__ float lane_xor(float v, int lane) {
    if constexpr (MASK < 32) {
        return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x1f | (MASK << 10)));
    } else {
        return __int_as_float(__builtin_amdgcn_ds_bpermute((lane ^ MASK) << 2, __float_as_int(v)));
    }
}

__device__ __forceinline__ float lane_read(float v, int src_lane_byte_addr) {
    return __int_as_float(__builtin_amdgcn_ds_bpermute(src_lane_byte_addr, __float_as_int(v)));
}

// Fallback naive attention kernel on GPU for arbitrary / unspecialized D:
__global__ void gfx803_fallback_attention_f32(
    const float* __restrict__ Q, // [B, H, N, D]
    const float* __restrict__ K, // [B, H, M, D]
    const float* __restrict__ V, // [B, H, M, D]
    float* __restrict__ O,       // [B, H, N, D]
    int B, int H, int N, int M, int D,
    float scale
) {
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    size_t q_base = ((size_t)b * H + h) * N * D + (size_t)i * D;
    size_t kv_head_base = ((size_t)b * H + h) * M * D;
    size_t o_base = q_base;

    float max_s = -1e20f;
    for (int j = 0; j < M; ++j) {
        float dot = 0.0f;
        size_t k_base = kv_head_base + (size_t)j * D;
        for (int d = 0; d < D; ++d) {
            dot += Q[q_base + d] * K[k_base + d];
        }
        dot *= scale;
        if (dot > max_s) max_s = dot;
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < M; ++j) {
        float dot = 0.0f;
        size_t k_base = kv_head_base + (size_t)j * D;
        for (int d = 0; d < D; ++d) {
            dot += Q[q_base + d] * K[k_base + d];
        }
        sum_exp += expf(dot * scale - max_s);
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < M; ++j) {
            float dot = 0.0f;
            size_t k_base = kv_head_base + (size_t)j * D;
            for (int kd = 0; kd < D; ++kd) {
                dot += Q[q_base + kd] * K[k_base + kd];
            }
            float p = expf(dot * scale - max_s) * inv_sum;
            size_t v_base = kv_head_base + (size_t)j * D;
            acc += p * V[v_base + d];
        }
        O[o_base + d] = acc;
    }
}

// ---------------------------------------------------------------------------
// Multi-wave fused attention for D = 40 / 64 / 80.
//
// One block = 4 waves = 128 query rows.  The block shares each 32-key K/V tile
// among its waves, so K and V stream from memory once per 128 rows instead of
// once per 32 rows.  Within a wave, lane (tx, ty) owns rows tx+8i and keys
// ty+8j of the 32x32 score tile (4x4 per lane) and accumulates P*V for its
// four rows over the d-slice [ty*D/8, (ty+1)*D/8).  The P values of the other
// key groups come through ds_bpermute, which saves a P store, a barrier and a
// re-read per tile, so the block needs only the Q, K and V tiles in LDS.
//
// LDS layout.  K and V rows are unpadded: the eight lanes of a key group read
// the same K address, and the eight lanes of a row group the same V address,
// which the LDS serves as a broadcast.  The Q row stride is D+4 floats, which
// keeps 16-byte alignment for ds_read_b128 and makes (stride / 16 B) odd, so
// the 8 consecutive rows one ds_read_b128 cycle serves fall into 8 distinct
// 16-byte bank groups.  Q*K^T reads 8 float4 per 64 FMAs.
//
// Every D runs one block (4 waves) per CU and prefetches the next K/V tile
// into registers while the current one is consumed.  D=40 would fit two
// blocks per CU in LDS (32768 B), but only within 128 VGPRs, and at that
// bound the compiler spills the accumulators of the v_mac_f32 loops (the
// destination is tied to the addend); the spilling build is 5x slower than
// the one-block, prefetching one.
// ---------------------------------------------------------------------------
// Element strides of one [B, H, rows, D] operand; D is contiguous.
struct AttnStrides { long long b, h, n; };

// Four consecutive elements of an operand: `load` keeps them in their storage
// type and `wide` converts to fp32. The K/V prefetch holds the raw values
// until the tile goes to LDS, because converting right after the load would
// make every tile wait for its loads instead of hiding them behind a tile of
// arithmetic. fp16 operands are one 8-byte load; gfx8 runs in unaligned mode
// under amdkfd.
template <typename E> struct AttnVec4;
template <> struct AttnVec4<float> {
    typedef float4 raw;
    static __device__ __forceinline__ raw load(const float* p) { return *reinterpret_cast<const float4*>(p); }
    static __device__ __forceinline__ raw zero() { return make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
    static __device__ __forceinline__ float4 wide(raw v) { return v; }
};
template <> struct AttnVec4<_Float16> {
    typedef _Float16 raw __attribute__((ext_vector_type(4)));
    static __device__ __forceinline__ raw load(const _Float16* p) { return *reinterpret_cast<const raw*>(p); }
    static __device__ __forceinline__ raw zero() { return raw{0, 0, 0, 0}; }
    static __device__ __forceinline__ float4 wide(raw v) { return make_float4(v.x, v.y, v.z, v.w); }
};

template <int D, int WAVES_PER_EU, bool PREFETCH, bool PIPE, typename E>
__global__ void __launch_bounds__(256, WAVES_PER_EU)
gfx803_fused_attention_mw_kernel(
    const E* __restrict__ Q, AttnStrides sq,   // [B, H, N, D]
    const E* __restrict__ K, AttnStrides sk,   // [B, H, M, D]
    const E* __restrict__ V, AttnStrides sv,   // [B, H, M, D]
    E* __restrict__ O, AttnStrides so,         // [B, H, N, D]
    int B, int H, int N, int M,
    float scale_log2,            // softmax scale * log2(e): exponentials use exp2
    float* __restrict__ LSE      // optional [B, H, N]: log2-sum-exp2 of the scaled scores, for backward
) {
    constexpr int NTHREADS = 256;
    constexpr int NWAVES = NTHREADS / WARP_SIZE;
    constexpr int BR_WAVE = 32;
    constexpr int BR = NWAVES * BR_WAVE;      // 128 query rows per block
    constexpr int BC = 32;                    // keys per tile
    constexpr int NG = 8;                     // row groups == key groups per wave
    constexpr int R = BR_WAVE / NG;           // rows per lane
    constexpr int C = BC / NG;                // keys per lane
    constexpr int DS = D / NG;                // d-slice per lane in P*V
    constexpr int D4 = D / 4;
    constexpr int QS = D + 4;                 // Q row stride in floats
    constexpr int KV_VEC = BC * D4;           // float4 per K (or V) tile
    constexpr int KV_PER_THREAD = (KV_VEC + NTHREADS - 1) / NTHREADS;
    constexpr float MASKED = -1e30f;

    static_assert(D % 4 == 0, "D must be a multiple of 4 for float4 traffic");
    static_assert(D % NG == 0, "D must split into 8 equal d-slices");
    static_assert(NG * NG == WARP_SIZE, "lane tiling must cover the 32x32 wave tile");
    static_assert(((QS / 4) & 1) == 1, "Q stride/16B must be odd for conflict-free ds_read_b128");
    static_assert(BR * QS * 4 + 2 * BC * D * 4 <= 65536, "LDS footprint exceeds the 64 KiB of a CU");

    __shared__ __attribute__((aligned(16))) float s_Q[BR * QS];
    __shared__ __attribute__((aligned(16))) float s_K[BC * D];
    __shared__ __attribute__((aligned(16))) float s_V[BC * D];

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & (WARP_SIZE - 1);
    const int tx = lane & (NG - 1);
    const int ty = lane >> 3;

    const int b = blockIdx.z / H;
    const int h = blockIdx.z % H;
    const int q0 = blockIdx.x * BR;

    const E* __restrict__ Qh = Q + b * sq.b + h * sq.h;
    const E* __restrict__ Kh = K + b * sk.b + h * sk.h;
    const E* __restrict__ Vh = V + b * sv.b + h * sv.h;

    // Q enters LDS pre-multiplied by scale*log2(e) so the inner loops never
    // touch the scale and the exponentials are plain exp2.
    for (int v = tid; v < BR * D4; v += NTHREADS) {
        const int r = v / D4;
        const int d4 = v - r * D4;
        float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (q0 + r < N) {
            val = AttnVec4<E>::wide(AttnVec4<E>::load(Qh + (q0 + r) * sq.n + d4 * 4));
            val.x *= scale_log2; val.y *= scale_log2; val.z *= scale_log2; val.w *= scale_log2;
        }
        *reinterpret_cast<float4*>(&s_Q[r * QS + d4 * 4]) = val;
    }

    typename AttnVec4<E>::raw k_reg[KV_PER_THREAD];
    typename AttnVec4<E>::raw v_reg[KV_PER_THREAD];

    // Rows past M stay zero so that P (== 0 there) never multiplies garbage
    // that could be NaN.
    auto load_tile = [&](int kt) {
        const int k0 = kt * BC;
        #pragma unroll
        for (int u = 0; u < KV_PER_THREAD; ++u) {
            const int v = tid + u * NTHREADS;
            k_reg[u] = AttnVec4<E>::zero();
            v_reg[u] = AttnVec4<E>::zero();
            if (v < KV_VEC) {
                const int r = v / D4;
                const int d4 = v - r * D4;
                if (k0 + r < M) {
                    k_reg[u] = AttnVec4<E>::load(Kh + (k0 + r) * sk.n + d4 * 4);
                    v_reg[u] = AttnVec4<E>::load(Vh + (k0 + r) * sv.n + d4 * 4);
                }
            }
        }
    };
    auto store_tile = [&]() {
        #pragma unroll
        for (int u = 0; u < KV_PER_THREAD; ++u) {
            const int v = tid + u * NTHREADS;
            if (v < KV_VEC) {
                const int r = v / D4;
                const int d4 = v - r * D4;
                *reinterpret_cast<float4*>(&s_K[r * D + d4 * 4]) = AttnVec4<E>::wide(k_reg[u]);
                *reinterpret_cast<float4*>(&s_V[r * D + d4 * 4]) = AttnVec4<E>::wide(v_reg[u]);
            }
        }
    };

    float m_run[R];
    float l_run[R];
    float acc[R][DS];
    #pragma unroll
    for (int i = 0; i < R; ++i) {
        m_run[i] = MASKED;
        l_run[i] = 0.0f;
        #pragma unroll
        for (int k = 0; k < DS; ++k) acc[i][k] = 0.0f;
    }

    const float* s_Qw = s_Q + wave * BR_WAVE * QS;
    const int num_tiles = (M + BC - 1) / BC;

    load_tile(0);
    for (int kt = 0; kt < num_tiles; ++kt) {
        store_tile();
        __syncthreads();
        if (PREFETCH && kt + 1 < num_tiles) load_tile(kt + 1);

        float S[R][C];
        #pragma unroll
        for (int i = 0; i < R; ++i)
            #pragma unroll
            for (int j = 0; j < C; ++j) S[i][j] = 0.0f;

        // Q*K^T.  The fragments of one d4 step are fetched as a group and,
        // with PIPE, the next step's group is in flight while this one is
        // consumed; the scheduling barrier keeps the compiler from sinking
        // each load to just before its first use, which would expose the LDS
        // latency once per 16 FMAs.
        float4 qa[R], ka[C], qb[R], kb[C];
        auto load_qk = [&](int d4, float4 (&q)[R], float4 (&k)[C]) {
            #pragma unroll
            for (int i = 0; i < R; ++i)
                q[i] = *reinterpret_cast<const float4*>(&s_Qw[(tx + NG * i) * QS + d4 * 4]);
            #pragma unroll
            for (int j = 0; j < C; ++j)
                k[j] = *reinterpret_cast<const float4*>(&s_K[(ty + NG * j) * D + d4 * 4]);
        };
        // Plain a * b + c, never fmaf(): gfx803 issues the fused v_fma_f32 at
        // a quarter of the rate of v_mac_f32 / v_mad_f32, which the compiler
        // contracts a * b + c into because fp32 denormals are flushed on gfx8.
        // fmaf() here made both GEMM loops of this kernel 2-4x slower.
        auto fma_qk = [&](const float4 (&q)[R], const float4 (&k)[C]) {
            #pragma unroll
            for (int j = 0; j < C; ++j)
                #pragma unroll
                for (int i = 0; i < R; ++i) {
                    S[i][j] += q[i].x * k[j].x;
                    S[i][j] += q[i].y * k[j].y;
                    S[i][j] += q[i].z * k[j].z;
                    S[i][j] += q[i].w * k[j].w;
                }
        };
        if (PIPE) {
            load_qk(0, qa, ka);
            #pragma unroll
            for (int d4 = 0; d4 < D4; d4 += 2) {
                if (d4 + 1 < D4) load_qk(d4 + 1, qb, kb);
                SCHED_BARRIER();
                fma_qk(qa, ka);
                if (d4 + 1 < D4) {
                    if (d4 + 2 < D4) load_qk(d4 + 2, qa, ka);
                    SCHED_BARRIER();
                    fma_qk(qb, kb);
                }
            }
        } else {
            #pragma unroll
            for (int d4 = 0; d4 < D4; ++d4) {
                load_qk(d4, qa, ka);
                SCHED_BARRIER();
                fma_qk(qa, ka);
            }
        }

        const int k0 = kt * BC;
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            float m_tile = MASKED;
            #pragma unroll
            for (int j = 0; j < C; ++j) {
                if (k0 + ty + NG * j >= M) S[i][j] = MASKED;
                m_tile = fmaxf(m_tile, S[i][j]);
            }
            m_tile = fmaxf(m_tile, lane_xor<NG>(m_tile, lane));
            m_tile = fmaxf(m_tile, lane_xor<2 * NG>(m_tile, lane));
            m_tile = fmaxf(m_tile, lane_xor<4 * NG>(m_tile, lane));

            // Every tile holds at least one key < M, so m_new is finite and
            // exp2 of (MASKED - m_new) underflows to exactly 0 for masked keys
            // and for the initial m_run.
            const float m_new = fmaxf(m_run[i], m_tile);
            const float alpha = exp2f(m_run[i] - m_new);
            float p_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < C; ++j) {
                S[i][j] = exp2f(S[i][j] - m_new);
                p_sum += S[i][j];
            }
            p_sum += lane_xor<NG>(p_sum, lane);
            p_sum += lane_xor<2 * NG>(p_sum, lane);
            p_sum += lane_xor<4 * NG>(p_sum, lane);

            l_run[i] = l_run[i] * alpha + p_sum;
            m_run[i] = m_new;
            #pragma unroll
            for (int k = 0; k < DS; ++k) acc[i][k] *= alpha;
        }

        // P*V over all 32 keys.  Key g+8j lives in lane (tx, g) as S[i][j];
        // keys are walked as n = 4g + j, so g and j are compile-time after
        // unrolling and the bpermute source register is a fixed VGPR.  The
        // P values and V slice of key n+1 are fetched while key n is consumed.
        float pa[R], va[DS], pb[R], vb[DS];
        auto load_pv = [&](int n, float (&p)[R], float (&v)[DS]) {
            const int g = n >> 2;
            const int j = n & 3;
            #pragma unroll
            for (int i = 0; i < R; ++i) p[i] = lane_read(S[i][j], (tx << 2) + (NG * 4) * g);
            const float* v_row = &s_V[(g + NG * j) * D + ty * DS];
            #pragma unroll
            for (int k = 0; k < DS; ++k) v[k] = v_row[k];
        };
        auto fma_pv = [&](const float (&p)[R], const float (&v)[DS]) {
            #pragma unroll
            for (int k = 0; k < DS; ++k)
                #pragma unroll
                for (int i = 0; i < R; ++i) acc[i][k] += p[i] * v[k];
        };
        load_pv(0, pa, va);
        #pragma unroll
        for (int n = 0; n < BC; n += 2) {
            load_pv(n + 1, pb, vb);
            SCHED_BARRIER();
            fma_pv(pa, va);
            if (n + 2 < BC) load_pv(n + 2, pa, va);
            SCHED_BARRIER();
            fma_pv(pb, vb);
        }
        __syncthreads();
        if (!PREFETCH && kt + 1 < num_tiles) load_tile(kt + 1);
    }

    #pragma unroll
    for (int i = 0; i < R; ++i) {
        const int q_global = q0 + wave * BR_WAVE + tx + NG * i;
        if (q_global >= N) continue;
        const float inv_l = (l_run[i] > 0.0f) ? (1.0f / l_run[i]) : 0.0f;
        if (LSE && ty == 0) LSE[((size_t)b * H + h) * N + q_global] = m_run[i] + log2f(l_run[i]);
        E* o_row = O + b * so.b + h * so.h + q_global * so.n + ty * DS;
        #pragma unroll
        for (int k = 0; k < DS; ++k) o_row[k] = static_cast<E>(acc[i][k] * inv_l);
    }
}

// Per-D configuration: waves per EU the compiler must fit, next-tile prefetch.
// The macros exist so a build can try another configuration without editing
// the source; the defaults are the shipped ones.
#ifndef ATTN2_WPE40
#define ATTN2_WPE40 1
#endif
#ifndef ATTN2_PF40
#define ATTN2_PF40 1
#endif
#ifndef ATTN2_WPE64
#define ATTN2_WPE64 1
#endif
#ifndef ATTN2_PF64
#define ATTN2_PF64 1
#endif
#ifndef ATTN2_WPE80
#define ATTN2_WPE80 1
#endif
#ifndef ATTN2_PF80
#define ATTN2_PF80 1
#endif
#ifndef ATTN2_PIPE40
#define ATTN2_PIPE40 1
#endif
#ifndef ATTN2_PIPE64
#define ATTN2_PIPE64 1
#endif
#ifndef ATTN2_PIPE80
#define ATTN2_PIPE80 1
#endif
template <int D> struct MwConfig;
template <> struct MwConfig<40> { static constexpr int WPE = ATTN2_WPE40; static constexpr bool PF = ATTN2_PF40; static constexpr bool PIPE = ATTN2_PIPE40; };
template <> struct MwConfig<64> { static constexpr int WPE = ATTN2_WPE64; static constexpr bool PF = ATTN2_PF64; static constexpr bool PIPE = ATTN2_PIPE64; };
template <> struct MwConfig<80> { static constexpr int WPE = ATTN2_WPE80; static constexpr bool PF = ATTN2_PF80; static constexpr bool PIPE = ATTN2_PIPE80; };

template <int D, typename E>
static hipError_t launch_mw(const E* Q, AttnStrides sq, const E* K, AttnStrides sk, const E* V, AttnStrides sv,
                            E* O, AttnStrides so, int B, int H, int N, int M, float scale, hipStream_t stream,
                            float* LSE = nullptr) {
    using Cfg = MwConfig<D>;
    const float scale_log2 = scale * 1.4426950408889634f;
    dim3 block(256);
    dim3 grid((N + 127) / 128, 1, B * H);
    (void)hipGetLastError();
    gfx803_fused_attention_mw_kernel<D, Cfg::WPE, Cfg::PF, Cfg::PIPE, E><<<grid, block, 0, stream>>>(
        Q, sq, K, sk, V, sv, O, so, B, H, N, M, scale_log2, LSE);
    return hipGetLastError();
}

template <int D>
static hipError_t launch_mw(const float* Q, const float* K, const float* V, float* O,
                            int B, int H, int N, int M, float scale, hipStream_t stream) {
    const AttnStrides sq{(long long)H * N * D, (long long)N * D, D};
    const AttnStrides skv{(long long)H * M * D, (long long)M * D, D};
    return launch_mw<D, float>(Q, sq, K, skv, V, skv, O, sq, B, H, N, M, scale, stream);
}


// ---------------------------------------------------------------------------
// Previous generation kernels.  They stay in the library for D=160 (the 2-wave
// kernel) and as the reference path gfx803_fused_attention_forward_v1, so one
// process can compare both implementations on the same buffers.
// ---------------------------------------------------------------------------

// 2D Block-Tiled Fused Attention Kernel for gfx803:
// Tile: Br = 32 queries, Bc = 32 keys/values.
// 64 threads (1 wavefront) per thread block.
// Online softmax tracks running maximum and normalizer in registers.
template <int D>
__launch_bounds__(64, 1)
__global__ void gfx803_fused_attention_2d_block_kernel(
    const float* __restrict__ Q, // [B, H, N, D]
    const float* __restrict__ K, // [B, H, M, D]
    const float* __restrict__ V, // [B, H, M, D]
    float* __restrict__ O,       // [B, H, N, D]
    int B, int H, int N, int M,
    float scale
) {
    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int D_PER_TY = D / 8; // e.g. 40/8=5, 64/8=8, 80/8=10
    constexpr int D_VEC = D / 4;

    const int tid = threadIdx.x; // 0 .. 63
    const int tx = tid & 7;      // 0 .. 7 (row group)
    const int ty = tid >> 3;     // 0 .. 7 (col group)

    const int b = blockIdx.z / H;
    const int h = blockIdx.z % H;
    const int q_tile_idx = blockIdx.x;

    const size_t q_head_base = ((size_t)b * H + h) * N * D;
    const size_t kv_head_base = ((size_t)b * H + h) * M * D;

    // Conflict-free padded LDS storage: stride has gcd(stride, 32) = 1
    __shared__ float s_Q[Br][D + 1];
    __shared__ float s_K[Bc][D + 1];
    __shared__ float s_V[Bc][D + 1];
    __shared__ float s_P[Br][Bc + 1];

    const int row_base = tx * 4;
    const int col_base = ty * 4;

    constexpr int TOTAL_Q_VEC = Br * D_VEC;
    const float4* Q_vec = reinterpret_cast<const float4*>(Q + q_head_base + (size_t)q_tile_idx * Br * D);
    for (int v_idx = tid; v_idx < TOTAL_Q_VEC; v_idx += 64) {
        int r = v_idx / D_VEC;
        int d4 = v_idx % D_VEC;
        int q_global = q_tile_idx * Br + r;
        if (q_global < N) {
            float4 q_val = Q_vec[v_idx];
            s_Q[r][d4 * 4 + 0] = q_val.x;
            s_Q[r][d4 * 4 + 1] = q_val.y;
            s_Q[r][d4 * 4 + 2] = q_val.z;
            s_Q[r][d4 * 4 + 3] = q_val.w;
        } else {
            s_Q[r][d4 * 4 + 0] = 0.0f;
            s_Q[r][d4 * 4 + 1] = 0.0f;
            s_Q[r][d4 * 4 + 2] = 0.0f;
            s_Q[r][d4 * 4 + 3] = 0.0f;
        }
    }

    float m_prev[4];
    float l_prev[4];
    float acc_O[4][D_PER_TY];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        m_prev[i] = -1e30f;
        l_prev[i] = 0.0f;
        #pragma unroll
        for (int kd = 0; kd < D_PER_TY; ++kd) {
            acc_O[i][kd] = 0.0f;
        }
    }

    const int num_k_tiles = (M + Bc - 1) / Bc;
    constexpr int TOTAL_KV_VEC = Bc * D_VEC;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const float4* K_vec = reinterpret_cast<const float4*>(K + kv_head_base + (size_t)kt * Bc * D);
        const float4* V_vec = reinterpret_cast<const float4*>(V + kv_head_base + (size_t)kt * Bc * D);
        for (int v_idx = tid; v_idx < TOTAL_KV_VEC; v_idx += 64) {
            int r = v_idx / D_VEC;
            int d4 = v_idx % D_VEC;
            int k_global = kt * Bc + r;
            if (k_global < M) {
                float4 k_val = K_vec[v_idx];
                float4 v_val = V_vec[v_idx];
                s_K[r][d4 * 4 + 0] = k_val.x;
                s_K[r][d4 * 4 + 1] = k_val.y;
                s_K[r][d4 * 4 + 2] = k_val.z;
                s_K[r][d4 * 4 + 3] = k_val.w;
                s_V[r][d4 * 4 + 0] = v_val.x;
                s_V[r][d4 * 4 + 1] = v_val.y;
                s_V[r][d4 * 4 + 2] = v_val.z;
                s_V[r][d4 * 4 + 3] = v_val.w;
            } else {
                s_K[r][d4 * 4 + 0] = 0.0f;
                s_K[r][d4 * 4 + 1] = 0.0f;
                s_K[r][d4 * 4 + 2] = 0.0f;
                s_K[r][d4 * 4 + 3] = 0.0f;
                s_V[r][d4 * 4 + 0] = 0.0f;
                s_V[r][d4 * 4 + 1] = 0.0f;
                s_V[r][d4 * 4 + 2] = 0.0f;
                s_V[r][d4 * 4 + 3] = 0.0f;
            }
        }
        __syncthreads();

        float S[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                S[i][j] = 0.0f;
            }
        }

        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            float q_val[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                q_val[i] = s_Q[row_base + i][d];
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float k_val = s_K[col_base + j][d];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    S[i][j] += q_val[i] * k_val;
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int k_global = kt * Bc + col_base + j;
                S[i][j] = (k_global < M) ? (S[i][j] * scale) : -1e30f;
            }
        }

        float row_m[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            row_m[i] = fmaxf(fmaxf(S[i][0], S[i][1]), fmaxf(S[i][2], S[i][3]));
            #pragma unroll
            for (int mask = 32; mask >= 8; mask >>= 1) {
                row_m[i] = fmaxf(row_m[i], __shfl_xor(row_m[i], mask));
            }
        }

        float P[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float m_curr = fmaxf(m_prev[i], row_m[i]);
            float alpha = (m_prev[i] == -1e30f) ? 0.0f : expf(m_prev[i] - m_curr);

            #pragma unroll
            for (int kd = 0; kd < D_PER_TY; ++kd) {
                acc_O[i][kd] *= alpha;
            }

            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                P[i][j] = (S[i][j] == -1e30f) ? 0.0f : expf(S[i][j] - m_curr);
                sum += P[i][j];
            }

            #pragma unroll
            for (int mask = 32; mask >= 8; mask >>= 1) {
                sum += __shfl_xor(sum, mask);
            }

            l_prev[i] = l_prev[i] * alpha + sum;
            m_prev[i] = m_curr;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                s_P[row_base + i][col_base + j] = P[i][j];
            }
        }
        __syncthreads();

        const int d_base = ty * D_PER_TY;
        #pragma unroll 4
        for (int j = 0; j < Bc; ++j) {
            float p_val[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                p_val[i] = s_P[row_base + i][j];
            }
            #pragma unroll
            for (int kd = 0; kd < D_PER_TY; ++kd) {
                float v_val = s_V[j][d_base + kd];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc_O[i][kd] += p_val[i] * v_val;
                }
            }
        }
        __syncthreads();
    }

    const int d_base = ty * D_PER_TY;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int q_global = q_tile_idx * Br + row_base + i;
        if (q_global < N) {
            float inv_l = (l_prev[i] > 0.0f) ? (1.0f / l_prev[i]) : 0.0f;
            size_t o_base = q_head_base + (size_t)q_global * D + d_base;
            #pragma unroll
            for (int kd = 0; kd < D_PER_TY; ++kd) {
                O[o_base + kd] = acc_O[i][kd] * inv_l;
            }
        }
    }
}

template __global__ void gfx803_fused_attention_2d_block_kernel<40>(const float*, const float*, const float*, float*, int, int, int, int, float);
template __global__ void gfx803_fused_attention_2d_block_kernel<64>(const float*, const float*, const float*, float*, int, int, int, int, float);

// 2-Wave Fused Attention Kernel for larger head dimensions (D=80, D=160):
// 128 threads (2 wavefronts: wave 0 handles rows 0..15, wave 1 handles rows 16..31).
template <int D, int Br, int Bc>
__launch_bounds__(128, 1)
__global__ void gfx803_fused_attention_2wave_kernel(
    const float* __restrict__ Q, // [B, H, N, D]
    const float* __restrict__ K, // [B, H, M, D]
    const float* __restrict__ V, // [B, H, M, D]
    float* __restrict__ O,       // [B, H, N, D]
    int B, int H, int N, int M,
    float scale
) {
    constexpr int D_PER_TY = D / 8;
    constexpr int D_VEC = D / 4;
    constexpr int ROWS_PER_THREAD = (Br / 2) / 8;
    constexpr int COLS_PER_THREAD = Bc / 8;

    const int tid = threadIdx.x;
    const int wave_id = tid >> 6;
    const int lane_id = tid & 63;
    const int tx = lane_id & 7;
    const int ty = lane_id >> 3;

    const int b = blockIdx.z / H;
    const int h = blockIdx.z % H;
    const int q_tile_idx = blockIdx.x;

    const size_t q_head_base = ((size_t)b * H + h) * N * D;
    const size_t kv_head_base = ((size_t)b * H + h) * M * D;

    __shared__ float s_Q[Br][D + 1];
    __shared__ float s_K[Bc][D + 1];
    __shared__ float s_V[Bc][D + 1];
    __shared__ float s_P[Br][Bc + 1];

    const int row_base = wave_id * (Br / 2) + tx * ROWS_PER_THREAD;
    const int col_base = ty * COLS_PER_THREAD;

    constexpr int TOTAL_Q_VEC = Br * D_VEC;
    const float4* Q_vec = reinterpret_cast<const float4*>(Q + q_head_base + (size_t)q_tile_idx * Br * D);
    for (int v_idx = tid; v_idx < TOTAL_Q_VEC; v_idx += 128) {
        int r = v_idx / D_VEC;
        int d4 = v_idx % D_VEC;
        int q_global = q_tile_idx * Br + r;
        if (q_global < N) {
            float4 q_val = Q_vec[v_idx];
            s_Q[r][d4 * 4 + 0] = q_val.x;
            s_Q[r][d4 * 4 + 1] = q_val.y;
            s_Q[r][d4 * 4 + 2] = q_val.z;
            s_Q[r][d4 * 4 + 3] = q_val.w;
        } else {
            s_Q[r][d4 * 4 + 0] = 0.0f;
            s_Q[r][d4 * 4 + 1] = 0.0f;
            s_Q[r][d4 * 4 + 2] = 0.0f;
            s_Q[r][d4 * 4 + 3] = 0.0f;
        }
    }

    float m_prev[ROWS_PER_THREAD];
    float l_prev[ROWS_PER_THREAD];
    float acc_O[ROWS_PER_THREAD][D_PER_TY];

    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        m_prev[i] = -1e30f;
        l_prev[i] = 0.0f;
        #pragma unroll
        for (int kd = 0; kd < D_PER_TY; ++kd) {
            acc_O[i][kd] = 0.0f;
        }
    }

    const int num_k_tiles = (M + Bc - 1) / Bc;
    constexpr int TOTAL_KV_VEC = Bc * D_VEC;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const float4* K_vec = reinterpret_cast<const float4*>(K + kv_head_base + (size_t)kt * Bc * D);
        const float4* V_vec = reinterpret_cast<const float4*>(V + kv_head_base + (size_t)kt * Bc * D);
        for (int v_idx = tid; v_idx < TOTAL_KV_VEC; v_idx += 128) {
            int r = v_idx / D_VEC;
            int d4 = v_idx % D_VEC;
            int k_global = kt * Bc + r;
            if (k_global < M) {
                float4 k_val = K_vec[v_idx];
                float4 v_val = V_vec[v_idx];
                s_K[r][d4 * 4 + 0] = k_val.x;
                s_K[r][d4 * 4 + 1] = k_val.y;
                s_K[r][d4 * 4 + 2] = k_val.z;
                s_K[r][d4 * 4 + 3] = k_val.w;
                s_V[r][d4 * 4 + 0] = v_val.x;
                s_V[r][d4 * 4 + 1] = v_val.y;
                s_V[r][d4 * 4 + 2] = v_val.z;
                s_V[r][d4 * 4 + 3] = v_val.w;
            } else {
                s_K[r][d4 * 4 + 0] = 0.0f;
                s_K[r][d4 * 4 + 1] = 0.0f;
                s_K[r][d4 * 4 + 2] = 0.0f;
                s_K[r][d4 * 4 + 3] = 0.0f;
                s_V[r][d4 * 4 + 0] = 0.0f;
                s_V[r][d4 * 4 + 1] = 0.0f;
                s_V[r][d4 * 4 + 2] = 0.0f;
                s_V[r][d4 * 4 + 3] = 0.0f;
            }
        }
        __syncthreads();

        float S[ROWS_PER_THREAD][COLS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            #pragma unroll
            for (int j = 0; j < COLS_PER_THREAD; ++j) {
                S[i][j] = 0.0f;
            }
        }

        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            float q_val[ROWS_PER_THREAD];
            #pragma unroll
            for (int i = 0; i < ROWS_PER_THREAD; ++i) {
                q_val[i] = s_Q[row_base + i][d];
            }
            #pragma unroll
            for (int j = 0; j < COLS_PER_THREAD; ++j) {
                float k_val = s_K[col_base + j][d];
                #pragma unroll
                for (int i = 0; i < ROWS_PER_THREAD; ++i) {
                    S[i][j] += q_val[i] * k_val;
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            #pragma unroll
            for (int j = 0; j < COLS_PER_THREAD; ++j) {
                int k_global = kt * Bc + col_base + j;
                S[i][j] = (k_global < M) ? (S[i][j] * scale) : -1e30f;
            }
        }

        float row_m[ROWS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            row_m[i] = S[i][0];
            #pragma unroll
            for (int j = 1; j < COLS_PER_THREAD; ++j) {
                row_m[i] = fmaxf(row_m[i], S[i][j]);
            }
            #pragma unroll
            for (int mask = 32; mask >= 8; mask >>= 1) {
                row_m[i] = fmaxf(row_m[i], __shfl_xor(row_m[i], mask));
            }
        }

        float P[ROWS_PER_THREAD][COLS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            float m_curr = fmaxf(m_prev[i], row_m[i]);
            float alpha = (m_prev[i] == -1e30f) ? 0.0f : expf(m_prev[i] - m_curr);

            #pragma unroll
            for (int kd = 0; kd < D_PER_TY; ++kd) {
                acc_O[i][kd] *= alpha;
            }

            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < COLS_PER_THREAD; ++j) {
                P[i][j] = (S[i][j] == -1e30f) ? 0.0f : expf(S[i][j] - m_curr);
                sum += P[i][j];
            }

            #pragma unroll
            for (int mask = 32; mask >= 8; mask >>= 1) {
                sum += __shfl_xor(sum, mask);
            }

            l_prev[i] = l_prev[i] * alpha + sum;
            m_prev[i] = m_curr;
        }

        #pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            #pragma unroll
            for (int j = 0; j < COLS_PER_THREAD; ++j) {
                s_P[row_base + i][col_base + j] = P[i][j];
            }
        }
        __syncthreads();

        const int d_base = ty * D_PER_TY;
        #pragma unroll 4
        for (int j = 0; j < Bc; ++j) {
            float p_val[ROWS_PER_THREAD];
            #pragma unroll
            for (int i = 0; i < ROWS_PER_THREAD; ++i) {
                p_val[i] = s_P[row_base + i][j];
            }
            #pragma unroll
            for (int kd = 0; kd < D_PER_TY; ++kd) {
                float v_val = s_V[j][d_base + kd];
                #pragma unroll
                for (int i = 0; i < ROWS_PER_THREAD; ++i) {
                    acc_O[i][kd] += p_val[i] * v_val;
                }
            }
        }
        __syncthreads();
    }

    const int d_base = ty * D_PER_TY;
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        int q_global = q_tile_idx * Br + row_base + i;
        if (q_global < N) {
            float inv_l = (l_prev[i] > 0.0f) ? (1.0f / l_prev[i]) : 0.0f;
            size_t o_base = q_head_base + (size_t)q_global * D + d_base;
            #pragma unroll
            for (int kd = 0; kd < D_PER_TY; ++kd) {
                O[o_base + kd] = acc_O[i][kd] * inv_l;
            }
        }
    }
}

template __global__ void gfx803_fused_attention_2wave_kernel<80, 32, 16>(const float*, const float*, const float*, float*, int, int, int, int, float);
template __global__ void gfx803_fused_attention_2wave_kernel<160, 32, 16>(const float*, const float*, const float*, float*, int, int, int, int, float);

static int launch_other(const float* Q, const float* K, const float* V, float* O,
                        int B, int H, int N, int M, int D, float scale, hipStream_t stream) {
    (void)hipGetLastError();
    if (D == 160) {
        dim3 block(128);
        dim3 grid((N + 31) / 32, 1, B * H);
        gfx803_fused_attention_2wave_kernel<160, 32, 16><<<grid, block, 0, stream>>>(Q, K, V, O, B, H, N, M, scale);
    } else {
        dim3 block(64);
        dim3 grid((N + 63) / 64, 1, B * H);
        gfx803_fallback_attention_f32<<<grid, block, 0, stream>>>(Q, K, V, O, B, H, N, M, D, scale);
    }
    return hipGetLastError() == hipSuccess ? 0 : -3;
}

static int check_args(const float* Q, const float* K, const float* V, const float* O,
                      int B, int H, int N, int M, int D) {
    if (Q == nullptr || K == nullptr || V == nullptr || O == nullptr) return -1;
    if (B <= 0 || H <= 0 || N <= 0 || M <= 0 || D <= 0) return -2;
    // All specialised kernels move 16-byte vectors; a misaligned base would
    // fault or read the neighbouring row instead of failing loudly.
    if (D % 4 == 0 && ((((uintptr_t)Q | (uintptr_t)K | (uintptr_t)V | (uintptr_t)O) & 15u) != 0)) return -4;
    return 0;
}

// ---------------------------------------------------------------------------
// Backward for training (fp32, contiguous [B, H, rows, D], head sizes 40/64/80).
// PyTorch's attention backward keeps the full score and probability matrices
// (2.6 GB extra for one SD1.5 self-attention at 512 px, batch 1), which is why
// 512 px LoRA training needs gradient checkpointing on an 8 GB card. These
// kernels recompute P from Q, K and the forward's per-row log-sum-exp instead,
// so memory grows with N, not N*M. dK/dV and dQ are two kernels, each
// parallel over its own tiles, because gfx803 has no global float atomic add.
// ---------------------------------------------------------------------------
namespace attn_bwd {

constexpr int BT = 32;   // keys or queries per tile

// S, P and dS for one 32x32 tile: thread t computes row t/8, columns 4*(t%8)..+3.
template <int D>
__device__ __forceinline__ void tile_p_ds(float (*sQ)[D + 1], float (*sK)[D + 1],
                                          float (*sdO)[D + 1], float (*sV)[D + 1],
                                          const float* s_lse, const float* s_dvec, float scale_log2,
                                          int qv, int kv, float (*sP)[BT + 1], float (*sdS)[BT + 1]) {
    const int t = threadIdx.x, q = t >> 3, k0 = (t & 7) * 4;
    float s[4] = {0, 0, 0, 0}, dp[4] = {0, 0, 0, 0};
    #pragma unroll 8
    for (int d = 0; d < D; ++d) {
        const float qd = sQ[q][d], gd = sdO[q][d];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            s[j] += qd * sK[k0 + j][d];
            dp[j] += gd * sV[k0 + j][d];
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const bool ok = q < qv && k0 + j < kv;
        const float p = ok ? exp2f(s[j] * scale_log2 - s_lse[q]) : 0.0f;
        sP[q][k0 + j] = p;
        sdS[q][k0 + j] = ok ? p * (dp[j] - s_dvec[q]) : 0.0f;
    }
}

template <int D>
__device__ __forceinline__ void load_rows(float (*dst)[D + 1], const float* src, int row0, int rows) {
    for (int i = threadIdx.x; i < BT * D; i += 256) {
        const int r = i / D, d = i - r * D;
        dst[r][d] = row0 + r < rows ? src[(size_t)(row0 + r) * D + d] : 0.0f;
    }
}


// dK = scale * dS^T Q, dV = P^T dO, one 32-key tile per block, looping over all query tiles.
template <int D>
__global__ void __launch_bounds__(256)
attn_bwd_dkdv(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
              const float* __restrict__ dO, const float* __restrict__ LSE, const float* __restrict__ DV,
              float* __restrict__ dK, float* __restrict__ dVo, int N, int M, float scale) {
    __shared__ float sQ[BT][D + 1], sK[BT][D + 1], sV[BT][D + 1], sdO[BT][D + 1];
    __shared__ float sP[BT][BT + 1], sdS[BT][BT + 1], s_lse[BT], s_dvec[BT];
    const size_t bh = blockIdx.y;
    const int k0 = blockIdx.x * BT, t = threadIdx.x;
    Q += bh * N * D; dO += bh * N * D; LSE += bh * N; DV += bh * N;
    K += bh * M * D; V += bh * M * D; dK += bh * M * D; dVo += bh * M * D;
    const float scale_log2 = scale * 1.4426950408889634f;
    load_rows<D>(sK, K, k0, M);
    load_rows<D>(sV, V, k0, M);
    constexpr int PER = D / 8;
    const int kk = t >> 3, dd = t & 7;
    float aK[PER], aV[PER];
    #pragma unroll
    for (int j = 0; j < PER; ++j) aK[j] = aV[j] = 0.0f;
    const int kv = M - k0 < BT ? M - k0 : BT;
    for (int q0 = 0; q0 < N; q0 += BT) {
        __syncthreads();
        load_rows<D>(sQ, Q, q0, N);
        load_rows<D>(sdO, dO, q0, N);
        if (t < BT) {
            s_lse[t] = q0 + t < N ? LSE[q0 + t] : 0.0f;
            s_dvec[t] = q0 + t < N ? DV[q0 + t] : 0.0f;
        }
        __syncthreads();
        const int qv = N - q0 < BT ? N - q0 : BT;
        tile_p_ds<D>(sQ, sK, sdO, sV, s_lse, s_dvec, scale_log2, qv, kv, sP, sdS);
        __syncthreads();
        #pragma unroll 4
        for (int q = 0; q < BT; ++q) {
            const float p = sP[q][kk], ds = sdS[q][kk];
            #pragma unroll
            for (int j = 0; j < PER; ++j) {
                aV[j] += p * sdO[q][dd + 8 * j];
                aK[j] += ds * sQ[q][dd + 8 * j];
            }
        }
    }
    if (k0 + kk < M) {
        #pragma unroll
        for (int j = 0; j < PER; ++j) {
            dK[(size_t)(k0 + kk) * D + dd + 8 * j] = aK[j] * scale;
            dVo[(size_t)(k0 + kk) * D + dd + 8 * j] = aV[j];
        }
    }
}

// dQ = scale * dS K, one 32-query tile per block, looping over all key tiles.
template <int D>
__global__ void __launch_bounds__(256)
attn_bwd_dq(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
            const float* __restrict__ dO, const float* __restrict__ LSE, const float* __restrict__ DV,
            float* __restrict__ dQ, int N, int M, float scale) {
    __shared__ float sQ[BT][D + 1], sK[BT][D + 1], sV[BT][D + 1], sdO[BT][D + 1];
    __shared__ float sP[BT][BT + 1], sdS[BT][BT + 1], s_lse[BT], s_dvec[BT];
    const size_t bh = blockIdx.y;
    const int q0 = blockIdx.x * BT, t = threadIdx.x;
    Q += bh * N * D; dO += bh * N * D; LSE += bh * N; DV += bh * N; dQ += bh * N * D;
    K += bh * M * D; V += bh * M * D;
    const float scale_log2 = scale * 1.4426950408889634f;
    load_rows<D>(sQ, Q, q0, N);
    load_rows<D>(sdO, dO, q0, N);
    if (t < BT) {
        s_lse[t] = q0 + t < N ? LSE[q0 + t] : 0.0f;
        s_dvec[t] = q0 + t < N ? DV[q0 + t] : 0.0f;
    }
    constexpr int PER = D / 8;
    const int qq = t >> 3, dd = t & 7;
    float aQ[PER];
    #pragma unroll
    for (int j = 0; j < PER; ++j) aQ[j] = 0.0f;
    const int qv = N - q0 < BT ? N - q0 : BT;
    for (int k0 = 0; k0 < M; k0 += BT) {
        __syncthreads();
        load_rows<D>(sK, K, k0, M);
        load_rows<D>(sV, V, k0, M);
        __syncthreads();
        const int kv = M - k0 < BT ? M - k0 : BT;
        tile_p_ds<D>(sQ, sK, sdO, sV, s_lse, s_dvec, scale_log2, qv, kv, sP, sdS);
        __syncthreads();
        #pragma unroll 4
        for (int k = 0; k < BT; ++k) {
            const float ds = sdS[qq][k];
            #pragma unroll
            for (int j = 0; j < PER; ++j) aQ[j] += ds * sK[k][dd + 8 * j];
        }
    }
    if (q0 + qq < N) {
        #pragma unroll
        for (int j = 0; j < PER; ++j) dQ[(size_t)(q0 + qq) * D + dd + 8 * j] = aQ[j] * scale;
    }
}

}  // namespace attn_bwd

extern "C" {

int gfx803_fused_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B, int H, int N, int M, int D,
    float scale,
    hipStream_t stream
) {
    int rc = check_args(Q, K, V, O, B, H, N, M, D);
    if (rc != 0) return rc;

    hipError_t err;
    if (D == 40) {
        err = launch_mw<40>(Q, K, V, O, B, H, N, M, scale, stream);
    } else if (D == 64) {
        err = launch_mw<64>(Q, K, V, O, B, H, N, M, scale, stream);
    } else if (D == 80) {
        err = launch_mw<80>(Q, K, V, O, B, H, N, M, scale, stream);
    } else {
        return launch_other(Q, K, V, O, B, H, N, M, D, scale, stream);
    }
    return err == hipSuccess ? 0 : -3;
}

// Q, K, V and O in any [B, H, rows, D] layout whose D is contiguous, given as
// element strides {b, h, n} per tensor, all four fp32 (dtype 0) or fp16
// (dtype 1). Reading ComfyUI's [B, rows, H*D] projections in place saves the
// transpose, the fp32 cast and the output copy that the contiguous fp32 entry
// point needs around every call. Head sizes 40, 64 and 80 only; anything else
// returns -4 for the caller to take another path.
int gfx803_fused_attention_forward_strided(
    const void* Q, const void* K, const void* V, void* O, int dtype,
    int B, int H, int N, int M, int D, float scale,
    const long long* strides,   // q b,h,n  k b,h,n  v b,h,n  o b,h,n
    hipStream_t stream
) {
    if (!Q || !K || !V || !O || !strides || B <= 0 || H <= 0 || N <= 0 || M <= 0) return -1;
    if ((D != 40 && D != 64 && D != 80) || (dtype != 0 && dtype != 1)) return -4;
    const AttnStrides sq{strides[0], strides[1], strides[2]}, sk{strides[3], strides[4], strides[5]};
    const AttnStrides sv{strides[6], strides[7], strides[8]}, so{strides[9], strides[10], strides[11]};
    hipError_t err;
#define GFX803_MW(DD, E) err = launch_mw<DD, E>((const E*)Q, sq, (const E*)K, sk, (const E*)V, sv, (E*)O, so, B, H, N, M, scale, stream)
    if (dtype == 0) {
        if (D == 40) GFX803_MW(40, float); else if (D == 64) GFX803_MW(64, float); else GFX803_MW(80, float);
    } else {
        if (D == 40) GFX803_MW(40, _Float16); else if (D == 64) GFX803_MW(64, _Float16); else GFX803_MW(80, _Float16);
    }
#undef GFX803_MW
    return err == hipSuccess ? 0 : -3;
}

// Training forward: contiguous fp32, also writes LSE [B, H, N] for the backward.
int gfx803_fused_attention_train_fwd(const float* Q, const float* K, const float* V, float* O, float* LSE,
                                     int B, int H, int N, int M, int D, float scale, hipStream_t stream) {
    int rc = check_args(Q, K, V, O, B, H, N, M, D);
    if (rc != 0) return rc;
    if (!LSE || (D != 40 && D != 64 && D != 80)) return -4;
    const AttnStrides sq{(long long)H * N * D, (long long)N * D, D}, skv{(long long)H * M * D, (long long)M * D, D};
    hipError_t err;
    if (D == 40) err = launch_mw<40, float>(Q, sq, K, skv, V, skv, O, sq, B, H, N, M, scale, stream, LSE);
    else if (D == 64) err = launch_mw<64, float>(Q, sq, K, skv, V, skv, O, sq, B, H, N, M, scale, stream, LSE);
    else err = launch_mw<80, float>(Q, sq, K, skv, V, skv, O, sq, B, H, N, M, scale, stream, LSE);
    return err == hipSuccess ? 0 : -3;
}

// Training backward: dQ, dK, dV from Q, K, V, dO, the forward's LSE and
// Dvec = rowsum(dO * O), all contiguous fp32.
int gfx803_fused_attention_train_bwd(const float* Q, const float* K, const float* V, const float* dO,
                                     const float* LSE, const float* Dvec, float* dQ, float* dK, float* dV,
                                     int B, int H, int N, int M, int D, float scale, hipStream_t stream) {
    if (!Q || !K || !V || !dO || !LSE || !Dvec || !dQ || !dK || !dV || B <= 0 || H <= 0 || N <= 0 || M <= 0) return -1;
    if (D != 40 && D != 64 && D != 80) return -4;
    using namespace attn_bwd;
    dim3 gk((M + BT - 1) / BT, B * H), gq((N + BT - 1) / BT, B * H);
    (void)hipGetLastError();
#define GFX803_BWD(DD) do { \
        attn_bwd_dkdv<DD><<<gk, 256, 0, stream>>>(Q, K, V, dO, LSE, Dvec, dK, dV, N, M, scale); \
        attn_bwd_dq<DD><<<gq, 256, 0, stream>>>(Q, K, V, dO, LSE, Dvec, dQ, N, M, scale); } while (0)
    if (D == 40) GFX803_BWD(40); else if (D == 64) GFX803_BWD(64); else GFX803_BWD(80);
#undef GFX803_BWD
    return hipGetLastError() == hipSuccess ? 0 : -3;
}

// Previous implementation (one wave per 32 query rows), kept for side-by-side
// checks of results and timing in the same process.
int gfx803_fused_attention_forward_v1(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B, int H, int N, int M, int D,
    float scale,
    hipStream_t stream
) {
    int rc = check_args(Q, K, V, O, B, H, N, M, D);
    if (rc != 0) return rc;

    (void)hipGetLastError();
    if (D == 40) {
        dim3 block(64);
        dim3 grid((N + 31) / 32, 1, B * H);
        gfx803_fused_attention_2d_block_kernel<40><<<grid, block, 0, stream>>>(Q, K, V, O, B, H, N, M, scale);
    } else if (D == 64) {
        dim3 block(64);
        dim3 grid((N + 31) / 32, 1, B * H);
        gfx803_fused_attention_2d_block_kernel<64><<<grid, block, 0, stream>>>(Q, K, V, O, B, H, N, M, scale);
    } else if (D == 80) {
        dim3 block(128);
        dim3 grid((N + 31) / 32, 1, B * H);
        gfx803_fused_attention_2wave_kernel<80, 32, 16><<<grid, block, 0, stream>>>(Q, K, V, O, B, H, N, M, scale);
    } else {
        return launch_other(Q, K, V, O, B, H, N, M, D, scale, stream);
    }
    return hipGetLastError() == hipSuccess ? 0 : -3;
}

}
