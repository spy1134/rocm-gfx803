#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <type_traits>

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

// The value of lane ^ 2. A DPP quad permutation: no LDS round trip.
__device__ __forceinline__ float lane_xor2(float v) {
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4e, 0xf, 0xf, false));
}

// acc += (p of the lane 2*S further along this lane's 16-lane row) * v, with
// the row rotation applied to the operand by the multiply-add itself. The
// compiler emits a separate v_mov_b32_dpp for __builtin_amdgcn_mov_dpp
// instead of folding it into v_mac_f32, hence the asm.
template <int S>
__device__ __forceinline__ void mac_row_rot2(float& acc, float p, float v) {
    if constexpr (S == 0)
        acc += p * v;
    else
        asm("v_mac_f32_dpp %0, %1, %2 row_ror:%3 row_mask:0xf bank_mask:0xf"
            : "+v"(acc) : "v"(p), "v"(v), "n"(16 - 2 * S));
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
// once per 32 rows, and the block needs only the Q, K and V tiles in LDS.
//
// Lane tiling.  Lane l of a wave owns rows tx+8i (i < 4) of the wave's 32
// query rows, tx = 2*(l/16) + l%2, and keys ty+8j (j < 4) of the 32-key tile,
// ty = (l/2)%8: a 4x4 block of the score tile.  In P*V the lane accumulates
// its four rows over the d-slice [ty*D/8, (ty+1)*D/8), which needs the P
// values of all 32 keys of those rows.  They lie in the eight lanes of the
// same 16-lane row with the same l%2, so at step s the lane takes key group
// (ty+s)%8, whose P values the lane 2s further along holds, and its
// v_mac_f32_dpp reads them through a DPP row rotation with no instruction of
// its own.  With one wave per SIMD, every instruction besides the
// multiply-adds costs the wave an issue slot: fetching P with ds_bpermute
// (128 per tile, in a layout with row groups across the low lane bits)
// measured 10% of the kernel's time, and with v_mov_b32_dpp 12%.
//
// LDS layout.  Q, K and V rows have a stride of D+4 floats, which keeps
// 16-byte alignment for ds_read_b128 and makes (stride / 16 B) odd, so the
// eight distinct rows one read serves fall into distinct 16-byte bank groups.
// The eight key groups of a 16-lane row read eight K rows at once; unpadded,
// those share one bank group, and the kernel ran 9% slower.  Q*K^T reads 8
// float4 per 64 multiply-adds.
//
// Every D runs one block (4 waves) per CU and prefetches the next K/V tile
// into registers while the current one is consumed.  A second block per CU
// needs the kernel within 128 VGPRs, and at that bound the compiler spills
// the accumulators of the v_mac_f32 loops (the destination is tied to the
// addend); the spilling build of D=40 was 5x slower than the one-block,
// prefetching one.
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
    constexpr int QS = D + 4;                 // Q, K and V row stride in floats
    constexpr int KV_VEC = BC * D4;           // float4 per K (or V) tile
    constexpr int KV_PER_THREAD = (KV_VEC + NTHREADS - 1) / NTHREADS;
    constexpr float MASKED = -1e30f;

    static_assert(D % 4 == 0, "D must be a multiple of 4 for float4 traffic");
    static_assert(D % NG == 0, "D must split into 8 equal d-slices");
    static_assert(NG * NG == WARP_SIZE, "lane tiling must cover the 32x32 wave tile");
    static_assert(((QS / 4) & 1) == 1, "Q stride/16B must be odd for conflict-free ds_read_b128");
    static_assert((BR + 2 * BC) * QS * 4 <= 65536, "LDS footprint exceeds the 64 KiB of a CU");

    __shared__ __attribute__((aligned(16))) float s_Q[BR * QS];
    __shared__ __attribute__((aligned(16))) float s_K[BC * QS];
    __shared__ __attribute__((aligned(16))) float s_V[BC * QS];

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & (WARP_SIZE - 1);
    const int tx = 2 * (lane >> 4) + (lane & 1);
    const int ty = (lane >> 1) & (NG - 1);

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
                *reinterpret_cast<float4*>(&s_K[r * QS + d4 * 4]) = AttnVec4<E>::wide(k_reg[u]);
                *reinterpret_cast<float4*>(&s_V[r * QS + d4 * 4]) = AttnVec4<E>::wide(v_reg[u]);
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
                k[j] = *reinterpret_cast<const float4*>(&s_K[(ty + NG * j) * QS + d4 * 4]);
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
            m_tile = fmaxf(m_tile, lane_xor2(m_tile));
            m_tile = fmaxf(m_tile, lane_xor<4>(m_tile, lane));
            m_tile = fmaxf(m_tile, lane_xor<8>(m_tile, lane));

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
            p_sum += lane_xor2(p_sum);
            p_sum += lane_xor<4>(p_sum, lane);
            p_sum += lane_xor<8>(p_sum, lane);

            l_run[i] = l_run[i] * alpha + p_sum;
            m_run[i] = m_new;
            #pragma unroll
            for (int k = 0; k < DS; ++k) acc[i][k] *= alpha;
        }

        // P*V over all 32 keys, step n = 4s + j taking key ((ty+s)%8) + 8j
        // (see the lane tiling above).  The V slice of step n+1 is fetched
        // while step n is consumed.
        float va[DS], vb[DS];
        auto load_v = [&](int n, float (&v)[DS]) {
            const float* v_row = &s_V[(((ty + (n >> 2)) & (NG - 1)) + NG * (n & 3)) * QS + ty * DS];
            #pragma unroll
            for (int k = 0; k < DS; ++k) v[k] = v_row[k];
        };
        auto pv_step = [&](auto s_tag) {
            constexpr int ST = decltype(s_tag)::value;
            auto fma_pv = [&](int j, const float (&v)[DS]) {
                #pragma unroll
                for (int k = 0; k < DS; ++k)
                    #pragma unroll
                    for (int i = 0; i < R; ++i) mac_row_rot2<ST>(acc[i][k], S[i][j], v[k]);
            };
            #pragma unroll
            for (int j = 0; j < 4; j += 2) {
                const int n = 4 * ST + j;
                load_v(n + 1, vb);
                SCHED_BARRIER();
                fma_pv(j, va);
                if (n + 2 < BC) load_v(n + 2, va);
                SCHED_BARRIER();
                fma_pv(j + 1, vb);
            }
        };
        load_v(0, va);
        pv_step(std::integral_constant<int, 0>{});
        // gfx8 needs two wait states between a VALU write of a VGPR and a DPP
        // read of it, and the compiler does not see the DPP inside the asm.
        asm volatile("s_nop 1");
        pv_step(std::integral_constant<int, 1>{});
        pv_step(std::integral_constant<int, 2>{});
        pv_step(std::integral_constant<int, 3>{});
        pv_step(std::integral_constant<int, 4>{});
        pv_step(std::integral_constant<int, 5>{});
        pv_step(std::integral_constant<int, 6>{});
        pv_step(std::integral_constant<int, 7>{});
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
// 512 px LoRA training needs gradient checkpointing on an 8 GB card. This
// backward recomputes P from Q, K and the forward's per-row log-sum-exp, so its
// memory grows with N, not N*M.
//
// Fused attention backward for gfx803, head sizes 40/64/80, fp32, contiguous [B*H, rows, D].
// Inputs Q, K, V, dO, LSE (log2-domain log-sum-exp of the scaled scores per query row, as the
// forward writes it), Dvec = rowsum(dO * O). Outputs dQ, dK, dV.
//
// S = QK^T and dP = dO V^T are computed once per (query, key) pair and feed all three gradients:
// 5 GEMM-sized products per pair, where one kernel per gradient direction needs 7. A block owns
// BK keys, keeps K and V in LDS for its whole life, streams the queries in BQ-row tiles and
// keeps dK and dV in registers. dQ needs a sum over every key block, and gfx803 has no float
// atomic add, so each block writes dQ summed over its own BK keys to a slab of a scratch buffer
// and a second kernel adds the slabs in a fixed order (deterministic, and no CAS loops).
//
// Lane tiling follows the forward kernel (lane (tx, ty) = (lane & 7, lane >> 3) owns query rows
// tx + 8i and keys ty + 8j of its wave's tile) so that S and dP come from float4 LDS reads with
// one operand broadcast. The P and dS values another lane needs come through ds_bpermute from
// the owning lane's registers: P and dS tiles in LDS (32 KiB per block) do not fit next to the
// K, V, Q and dO tiles.
#define LAUNCH(KERNEL, GRID, BLOCK, STREAM, ...) KERNEL<<<GRID, BLOCK, 0, STREAM>>>(__VA_ARGS__)

// Raw buffer access: the descriptor lives in SGPRs and each access needs one 32-bit VGPR
// offset, where a flat access needs a 64-bit VGPR address per distinct pointer, and the
// compiler keeps those addresses live across the whole query loop next to the dK/dV
// accumulators. An offset at or past the descriptor's byte count reads 0 and drops writes,
// which is how rows past N are handled without branches.
typedef int attnb_i32x4 __attribute__((ext_vector_type(4)));
typedef float attnb_f32x4 __attribute__((ext_vector_type(4)));
typedef float attnb_f32x2 __attribute__((ext_vector_type(2)));
__device__ attnb_f32x4 attnb_bl4(attnb_i32x4 r, int vo, int so, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4f32");
__device__ float attnb_bl1(attnb_i32x4 r, int vo, int so, int aux) __asm("llvm.amdgcn.raw.buffer.load.f32");
__device__ void attnb_bs4(attnb_f32x4 v, attnb_i32x4 r, int vo, int so, int aux) __asm("llvm.amdgcn.raw.buffer.store.v4f32");
__device__ void attnb_bs2(attnb_f32x2 v, attnb_i32x4 r, int vo, int so, int aux) __asm("llvm.amdgcn.raw.buffer.store.v2f32");
__device__ void attnb_bs1(float v, attnb_i32x4 r, int vo, int so, int aux) __asm("llvm.amdgcn.raw.buffer.store.f32");
struct Buf { attnb_i32x4 r; };
// Byte count in dword 2, stride 0, data format 32 in dword 3 (gfx8 treats a zero format as
// an invalid descriptor). The range check covers the VGPR offset, not soffset, so all
// variable offsets go into the VGPR operand. Callers pass block-uniform values, but the 64-bit
// address arithmetic runs on the VALU on gfx8 and the compiler then cannot prove the result
// uniform; readfirstlane states it, otherwise every access becomes a waterfall loop.
__device__ __forceinline__ Buf make_buf(const void* base, unsigned bytes) {
    const unsigned long long a = reinterpret_cast<unsigned long long>(base);
    Buf b;
    b.r.x = __builtin_amdgcn_readfirstlane((int)(unsigned)a);
    b.r.y = __builtin_amdgcn_readfirstlane((int)((unsigned)(a >> 32) & 0xffffu));
    b.r.z = __builtin_amdgcn_readfirstlane((int)bytes);
    b.r.w = 0x00020000;
    return b;
}
__device__ __forceinline__ float4 bload4(Buf b, int off) {
    const attnb_f32x4 v = attnb_bl4(b.r, off, 0, 0);
    return make_float4(v.x, v.y, v.z, v.w);
}
__device__ __forceinline__ float bload1(Buf b, int off) { return attnb_bl1(b.r, off, 0, 0); }
__device__ __forceinline__ void bstore4(Buf b, int off, float4 v) {
    attnb_f32x4 x; x.x = v.x; x.y = v.y; x.z = v.z; x.w = v.w;
    attnb_bs4(x, b.r, off, 0, 0);
}
__device__ __forceinline__ void bstore2(Buf b, int off, float2 v) {
    attnb_f32x2 x; x.x = v.x; x.y = v.y;
    attnb_bs2(x, b.r, off, 0, 0);
}
__device__ __forceinline__ void bstore1(Buf b, int off, float v) { attnb_bs1(v, b.r, off, 0, 0); }
#include <math.h>
#include <stddef.h>
#include <stdint.h>


namespace attn_bwd {

template <bool B> struct Own { static constexpr bool value = B; };

// Chunk swizzle of a row: rows r..r+7 (r a multiple of 8) read at the same logical chunk fall
// into 8 distinct 16-byte bank groups. Depends on r mod 8 only, so a row's swizzle is fixed by
// the lane's tx or ty. The XOR stays inside the row because D/4 is a multiple of 2, 8 and 4.
template <int D> __device__ __forceinline__ int swz(int r) {
    return D == 40 ? ((r >> 2) & 1) : D == 64 ? (r & 7) : ((r >> 1) & 3);
}
// d-group of the dK/dV accumulators of lane tx, and of the dQ accumulators of lane ty. With
// the identity, under the swizzle above, the dV/dK slice reads of the lanes of a pass collide
// 2-way at every step for D=40 and D=80, and the dQ K-slice reads 2-way for D=64; these
// permutations, found by exhaustive search over the 8! orders against a model of the LDS
// banks, make all 8 steps of each conflict-free.
template <int D> __device__ __forceinline__ int dgroup(int t) {
    constexpr uint32_t tab = D == 40 ? 0x75436210u : D == 64 ? 0x75361420u : 0x65324710u;
    return (tab >> (4 * t)) & 7;
}
template <int D> __device__ __forceinline__ int dqgroup(int t) {
    constexpr uint32_t tab = D == 64 ? 0x75643120u : 0x76543210u;
    return (tab >> (4 * t)) & 7;
}
template <int D> __device__ __forceinline__ int pel(int e, int fr) { return (((e >> 2) ^ fr) << 2) | (e & 3); }

__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ void st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }

// The D/8 elements of d-group g: chunks g, g+8, ... and then D/8 mod 4 elements from the
// tail 32*(D/32) .. D-1, so every group has float4 reads for all but at most two elements.
template <int D>
__device__ __forceinline__ void load_slice(const float* row, int g, int fr, float (&o)[D / 8]) {
    constexpr int DG = D / 8, NFC = DG / 4, RM = DG % 4;
    #pragma unroll
    for (int c = 0; c < NFC; ++c) {
        const float4 v = ld4(row + pel<D>(4 * (g + 8 * c), fr));
        o[4 * c] = v.x; o[4 * c + 1] = v.y; o[4 * c + 2] = v.z; o[4 * c + 3] = v.w;
    }
    if constexpr (RM == 1) o[4 * NFC] = row[pel<D>(32 * NFC + g, fr)];
    if constexpr (RM == 2) {
        const float2 v = *reinterpret_cast<const float2*>(row + pel<D>(32 * NFC + 2 * g, fr));
        o[4 * NFC] = v.x; o[4 * NFC + 1] = v.y;
    }
}
template <int D>
__device__ __forceinline__ void store_slice(float* row, int g, int fr, const float (&o)[D / 8]) {
    constexpr int DG = D / 8, NFC = DG / 4, RM = DG % 4;
    #pragma unroll
    for (int c = 0; c < NFC; ++c)
        st4(row + pel<D>(4 * (g + 8 * c), fr), make_float4(o[4 * c], o[4 * c + 1], o[4 * c + 2], o[4 * c + 3]));
    if constexpr (RM == 1) row[pel<D>(32 * NFC + g, fr)] = o[4 * NFC];
    if constexpr (RM == 2)
        *reinterpret_cast<float2*>(row + pel<D>(32 * NFC + 2 * g, fr)) = make_float2(o[4 * NFC], o[4 * NFC + 1]);
}

template <int D>
__device__ __forceinline__ void bstore_slice(Buf b, int row_off, int g, const float (&o)[D / 8]) {
    constexpr int DG = D / 8, NFC = DG / 4, RM = DG % 4;
    #pragma unroll
    for (int c = 0; c < NFC; ++c)
        bstore4(b, row_off + 16 * (g + 8 * c), make_float4(o[4 * c], o[4 * c + 1], o[4 * c + 2], o[4 * c + 3]));
    if constexpr (RM == 1) bstore1(b, row_off + 4 * (32 * NFC + g), o[4 * NFC]);
    if constexpr (RM == 2) bstore2(b, row_off + 4 * (32 * NFC + 2 * g), make_float2(o[4 * NFC], o[4 * NFC + 1]));
}

// acc[i][j] = dot(a row 8i, b row 8j) for a 4 x JP lane tile. Plain a * b + c: gfx803 issues
// fmaf()'s v_fma_f32 at a quarter of the rate of the v_mac_f32 this contracts to. Two steps per
// iteration let the scheduler issue the second step's loads under the first step's FMAs; an
// explicit double buffer across iterations costs about 100 more VGPRs with this compiler and
// spills.
template <int D, int JP>
__device__ __forceinline__ void tile_dot(const float* a, const float* b, int fa, int fb, float (&acc)[4][JP]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < JP; ++j) acc[i][j] = 0.0f;
    #pragma unroll 2
    for (int c = 0; c < D / 4; ++c) {
        float4 x[4], y[JP];
        #pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = ld4(a + 8 * i * D + ((c ^ fa) << 2));
        #pragma unroll
        for (int j = 0; j < JP; ++j) y[j] = ld4(b + 8 * j * D + ((c ^ fb) << 2));
        #pragma unroll
        for (int j = 0; j < JP; ++j)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i][j] += x[i].x * y[j].x;
                acc[i][j] += x[i].y * y[j].y;
                acc[i][j] += x[i].z * y[j].z;
                acc[i][j] += x[i].w * y[j].w;
            }
    }
}

// Block = 4 waves = WQ query groups x WK key groups. Wave (wq, wk) handles 32 query rows x KT
// keys of each (BQ x BK) tile pair: 4 rows x KT/8 keys per lane in S/dP, KT/8 keys x D/8
// elements per lane in dK/dV, 4 rows x D/8 elements per lane in dQ.
//
// Lane l has row group tx = (l/2)%8 and key group ty = l%2 + 2*(l/16). dV/dK needs, for
// this lane's keys, the P and dS values of all 8 row groups; they lie in the 8 lanes of the
// same 16-lane row with the same l%2, so step m takes row group (tx+m)%8, whose values the
// lane 2m further along holds, and v_mac_f32_dpp reads them through a DPP row rotation with
// no instruction of its own (see mac_row_rot2). Exchanged with ds_bpermute, as dQ's key
// groups across rows still are, these were 224 of the kernel's 336 lane reads per tile
// pair.
template <int D, int KT, int WQ, int WK>
__global__ void __launch_bounds__(256, 1)
attn_bwd_main(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
              const float* __restrict__ dO, const float* __restrict__ LSE, const float* __restrict__ DV,
              float* __restrict__ dK, float* __restrict__ dVo, float* __restrict__ dQp, long long slab,
              int N, int M, float scale) {
    constexpr int D4 = D / 4, DG = D / 8, JN = KT / 8, JP = JN < 4 ? JN : 4, NP = JN / JP;
    constexpr int BK = KT * WK, BQ = 32 * WQ;
    constexpr int QV4 = BQ * D4, QPT = (QV4 + 255) / 256;
    static_assert(WQ * WK == 4, "four waves per block");
    static_assert(D % 8 == 0 && KT % 8 == 0, "lane tiling");
    static_assert(2 * (BK + BQ) * D * 4 <= 65536, "LDS");
    static_assert(WK * BQ <= 2 * BQ, "dQ reduction must fit in the Q/dO tiles");
    static_assert(WQ * BK <= 2 * (BK + BQ), "dK/dV reduction must fit in LDS");

    __shared__ __attribute__((aligned(16))) float smem[2 * (BK + BQ) * D];
    float* const sK = smem;
    float* const sV = sK + BK * D;
    float* const sQ = sV + BK * D;
    float* const sG = sQ + BQ * D;

    const int tid = threadIdx.x, w = tid >> 6, lane = tid & 63;
    const int tx = (lane >> 1) & 7, ty = (lane & 1) | ((lane >> 4) << 1);
    const int wq = w % WQ, wk = w / WQ;
    const int kb = blockIdx.x;
    const size_t bh = blockIdx.y;
    const int k0 = kb * BK;
    Q += bh * N * D; dO += bh * N * D; LSE += bh * N; DV += bh * N;
    K += bh * M * D; V += bh * M * D; dK += bh * M * D; dVo += bh * M * D;
    float* const dQs = dQp + (size_t)kb * slab + bh * N * D;
    const float sl = scale * 1.4426950408889634f;

    // Rows past M stay zero so that their S and dP are finite before masking.
    for (int v = tid; v < BK * D4; v += 256) {
        const int r = v / D4, c = v - r * D4;
        float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f), b = a;
        if (k0 + r < M) { a = ld4(K + (size_t)(k0 + r) * D + 4 * c); b = ld4(V + (size_t)(k0 + r) * D + 4 * c); }
        const int o = r * D + ((c ^ swz<D>(r)) << 2);
        st4(sK + o, a);
        st4(sV + o, b);
    }

    float4 qreg[QPT], greg[QPT];
    float lse_n[4], dv_n[4];
    const int qrow = wq * 32 + tx;              // tile row of this lane's i = 0
    const int krow = wk * KT + ty;              // block key of this lane's j = 0
    // Rows past N read as zero: their S and dP are 0, so P = 1 and dS = 0 there, and P only
    // meets their zero dO rows in dV. LSE and Dvec read as 0 for them, which keeps P finite.
    auto load_tile = [&](int q0) __attribute__((always_inline)) {
        const unsigned rem = (unsigned)(N - q0) * D * 4;
        const Buf bq = make_buf(Q + (size_t)q0 * D, rem), bg = make_buf(dO + (size_t)q0 * D, rem);
        #pragma unroll
        for (int u = 0; u < QPT; ++u) {
            const int v = tid + 256 * u;
            if (QV4 % 256 == 0 || v < QV4) {
                qreg[u] = bload4(bq, 16 * v);
                greg[u] = bload4(bg, 16 * v);
            }
        }
        const Buf bl = make_buf(LSE + q0, (unsigned)(N - q0) * 4), bd = make_buf(DV + q0, (unsigned)(N - q0) * 4);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            lse_n[i] = bload1(bl, 4 * (qrow + 8 * i));
            dv_n[i] = bload1(bd, 4 * (qrow + 8 * i));
        }
    };
    auto store_tile = [&]() __attribute__((always_inline)) {
        #pragma unroll
        for (int u = 0; u < QPT; ++u) {
            const int v = tid + 256 * u;
            if (QV4 % 256 == 0 || v < QV4) {
                const int r = v / D4, c = v - r * D4;
                const int o = r * D + ((c ^ swz<D>(r)) << 2);
                st4(sQ + o, qreg[u]);
                st4(sG + o, greg[u]);
            }
        }
    };

    unsigned kvalid = 0;
    #pragma unroll
    for (int j = 0; j < JN; ++j) kvalid |= (k0 + krow + 8 * j < M ? 1u : 0u) << j;
    const int fq = swz<D>(tx), fk = swz<D>(ty);
    const int gkv = dgroup<D>(tx), gq = dqgroup<D>(ty);

    float aK[JN][DG], aV[JN][DG];
    #pragma unroll
    for (int j = 0; j < JN; ++j)
        #pragma unroll
        for (int e = 0; e < DG; ++e) aK[j][e] = aV[j][e] = 0.0f;

    const int nqt = (N + BQ - 1) / BQ;
    load_tile(0);
    for (int qt = 0; qt < nqt; ++qt) {
        const int q0 = qt * BQ;
        if (qt > 0) __syncthreads();
        store_tile();
        float lse[4], dvv[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) { lse[i] = lse_n[i]; dvv[i] = dv_n[i]; }
        __syncthreads();

        // Assigned by the first dQ product rather than zeroed here: zeroed accumulators get
        // materialised before the S/dP products and hold their VGPRs through them.
        float aQ[4][DG];

        // The wave's KT keys go through S/dP, dV/dK and dQ in passes of 8*JP keys, so that
        // only a 4 x JP tile of S and of dP is live next to the dK/dV accumulators.
        #pragma unroll
        for (int h = 0; h < NP; ++h) {
            // S becomes P and G (dP) becomes dS in place.
            float S[4][JP], G[4][JP];
            tile_dot<D, JP>(sQ + qrow * D, sK + (krow + 8 * JP * h) * D, fq, fk, S);
            tile_dot<D, JP>(sG + qrow * D, sV + (krow + 8 * JP * h) * D, fq, fk, G);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < JP; ++j) {
                    const float p = (kvalid >> (JP * h + j)) & 1u ? exp2f(S[i][j] * sl - lse[i]) : 0.0f;
                    G[i][j] = p * (G[i][j] - dvv[i]);
                    S[i][j] = p;
                }

            // dV += P^T dO, dK += dS^T Q over the wave's 32 rows. Step m takes the rows of row
            // group (tx+m)%8, whose P and dS for this lane's keys sit in the same registers of
            // the lane 2m further along (see the lane tiling above).
            auto dvdk = [&](auto m_tag) __attribute__((always_inline)) {
                constexpr int MS = decltype(m_tag)::value;
                // Opaque to the compiler, so each step's lane-dependent addresses are computed
                // here and not hoisted out of the query loop, where the eight steps' sets stayed
                // live and pushed D=64 and D=80 into spilling inside the loop.
                int ts = (tx + MS) & 7;
                asm volatile("" : "+v"(ts));
                const int fr = swz<D>(ts);
                const float* xr = sQ + (wq * 32 + ts) * D;
                const float* gr = sG + (wq * 32 + ts) * D;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float xs[DG], gs[DG];
                    load_slice<D>(xr + 8 * i * D, gkv, fr, xs);
                    load_slice<D>(gr + 8 * i * D, gkv, fr, gs);
                    #pragma unroll
                    for (int j = 0; j < JP; ++j)
                        #pragma unroll
                        for (int e = 0; e < DG; ++e) {
                            mac_row_rot2<MS>(aV[JP * h + j][e], S[i][j], gs[e]);
                            mac_row_rot2<MS>(aK[JP * h + j][e], G[i][j], xs[e]);
                        }
                }
                SCHED_BARRIER();
            };
            dvdk(std::integral_constant<int, 0>{});
            // gfx8 needs two wait states between a VALU write of a VGPR and a DPP read of it,
            // and the compiler does not see the DPP inside the asm.
            asm volatile("s_nop 1");
            dvdk(std::integral_constant<int, 1>{});
            dvdk(std::integral_constant<int, 2>{});
            dvdk(std::integral_constant<int, 3>{});
            dvdk(std::integral_constant<int, 4>{});
            dvdk(std::integral_constant<int, 5>{});
            dvdk(std::integral_constant<int, 6>{});
            dvdk(std::integral_constant<int, 7>{});

            // The Q/dO registers are free from here on; the loads hide behind the last dQ pass.
            if (h == NP - 1 && qt + 1 < nqt) load_tile(q0 + BQ);

            // dQ += dS K over this pass's keys. Step s takes the keys of lane ty ^ s.
            auto dq = [&](auto own, int s) __attribute__((always_inline)) {
                const int ys = ty ^ s;
                const int src = ((ys & 1) | (tx << 1) | ((ys >> 1) << 4)) << 2;
                const int fr = swz<D>(ys);
                const float* kr = sK + (krow - ty + ys + 8 * JP * h) * D;
                auto fetch = [&](int j, float (&ks)[DG], float (&d)[4]) __attribute__((always_inline)) {
                    load_slice<D>(kr + 8 * j * D, gq, fr, ks);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if constexpr (decltype(own)::value) d[i] = G[i][j];
                        else d[i] = lane_read(G[i][j], src);
                    }
                };
                auto mac = [&](int j, const float (&ks)[DG], const float (&d)[4]) __attribute__((always_inline)) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        #pragma unroll
                        for (int e = 0; e < DG; ++e) {
                            if (decltype(own)::value && h == 0 && j == 0) aQ[i][e] = d[i] * ks[e];
                            else aQ[i][e] += d[i] * ks[e];
                        }
                };
                {
                    #pragma unroll
                    for (int j = 0; j < JP; ++j) {
                        float ks[DG], d[4];
                        fetch(j, ks, d);
                        mac(j, ks, d);
                    }
                }
            };
            dq(Own<true>{}, 0);
            #pragma unroll 1
            for (int s = 1; s < 8; ++s) dq(Own<false>{}, s);
        }

        const Buf bs = make_buf(dQs + (size_t)q0 * D, (unsigned)(N - q0) * D * 4);
        if constexpr (WK == 1) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) bstore_slice<D>(bs, 4 * (qrow + 8 * i) * D, gq, aQ[i]);
        } else {
            // The WK waves of a query group hold partial sums over different keys; add them in
            // the Q/dO tiles, which no wave reads after its dV/dK step.
            __syncthreads();
            float* const red = sQ;
            #pragma unroll
            for (int i = 0; i < 4; ++i) store_slice<D>(red + (wk * BQ + qrow + 8 * i) * D, gq, fq, aQ[i]);
            __syncthreads();
            #pragma unroll
            for (int u = 0; u < QPT; ++u) {
                const int v = tid + 256 * u;
                if (QV4 % 256 == 0 || v < QV4) {
                    const int r = v / D4, c = v - r * D4;
                    const int o = r * D + ((c ^ swz<D>(r)) << 2);
                    float4 a = ld4(red + o);
                    #pragma unroll
                    for (int w2 = 1; w2 < WK; ++w2) {
                        const float4 b = ld4(red + w2 * BQ * D + o);
                        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
                    }
                    bstore4(bs, 16 * v, a);
                }
            }
        }
    }

    // dK and dV: add the WQ query groups' partial sums, one operand at a time to fit in LDS.
    #pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
        __syncthreads();
        float* const red = smem;
        #pragma unroll
        for (int j = 0; j < JN; ++j)
            store_slice<D>(red + (wq * BK + krow + 8 * j) * D, gkv, fk, pass ? aV[j] : aK[j]);
        __syncthreads();
        float* const out = pass ? dVo : dK;
        const float mul = pass ? 1.0f : scale;
        for (int v = tid; v < BK * D4; v += 256) {
            const int r = v / D4, c = v - r * D4;
            const int o = r * D + ((c ^ swz<D>(r)) << 2);
            float4 a = ld4(red + o);
            #pragma unroll
            for (int u = 1; u < WQ; ++u) {
                const float4 b = ld4(red + u * BK * D + o);
                a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
            }
            if (k0 + r < M) st4(out + (size_t)(k0 + r) * D + 4 * c, make_float4(a.x * mul, a.y * mul, a.z * mul, a.w * mul));
        }
    }
}

// dQ = scale * sum over key blocks of the slabs, in slab order.
__global__ void __launch_bounds__(256)
attn_bwd_dq_reduce(const float* __restrict__ part, float* __restrict__ dQ, long long n4, long long slab,
                   int ns, float scale) {
    for (long long e = (long long)blockIdx.x * 256 + threadIdx.x; e < n4; e += (long long)gridDim.x * 256) {
        float4 a = ld4(part + 4 * e);
        #pragma unroll 8
        for (int s = 1; s < ns; ++s) {
            const float4 b = ld4(part + s * slab + 4 * e);
            a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        }
        st4(dQ + 4 * e, make_float4(a.x * scale, a.y * scale, a.z * scale, a.w * scale));
    }
}

// Slabs are BH*N*D floats plus 64: for the usual power-of-two N the bare size is a multiple of
// 8 KiB, which would put the same rows of all concurrently running blocks' slabs on one DRAM
// channel.
inline long long slab_floats(int BH, int N, int D) { return (long long)BH * N * D + 64; }

template <int D, int KT, int WQ, int WK>
size_t ws_bytes(int BH, int N, int M) {
    const int ns = (M + KT * WK - 1) / (KT * WK);
    return (size_t)ns * slab_floats(BH, N, D) * sizeof(float);
}

template <int D, int KT, int WQ, int WK>
int run(const float* Q, const float* K, const float* V, const float* dO, const float* LSE, const float* DV,
        float* dQ, float* dK, float* dV, int BH, int N, int M, float scale, float* ws, hipStream_t stream) {
    constexpr int BK = KT * WK;
    const int ns = (M + BK - 1) / BK;
    const long long slab = slab_floats(BH, N, D);
    (void)hipGetLastError();
    auto kmain = &attn_bwd_main<D, KT, WQ, WK>;
    LAUNCH(kmain, dim3(ns, BH), dim3(256), stream, Q, K, V, dO, LSE, DV, dK, dV, ws, slab, N, M, scale);
    const long long n4 = (long long)BH * N * D / 4;
    const long long blocks = (n4 + 255) / 256;
    auto kred = &attn_bwd_dq_reduce;
    LAUNCH(kred, dim3((unsigned)blocks), dim3(256), stream, ws, dQ, n4, slab, ns, scale);
    return hipGetLastError() == hipSuccess ? 0 : -3;
}


// Default geometry per head size: 2 x 2 waves. 64 keys per block for D=40 and D=64 (dQ slabs
// per 64 keys); D=80 takes 32 keys per block because 64 keys and 64 query rows of K, V, Q and dO
// exceed 64 KiB.
template <int D> struct Geo;
template <> struct Geo<40> { static constexpr int KT = 32, WQ = 2, WK = 2; };
template <> struct Geo<64> { static constexpr int KT = 32, WQ = 2, WK = 2; };
template <> struct Geo<80> { static constexpr int KT = 16, WQ = 2, WK = 2; };

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

// Scratch the training backward needs, in bytes: the dQ partial sums of every
// key block. The caller owns it so that a framework's allocator can serve it.
size_t gfx803_fused_attention_train_ws_bytes(int B, int H, int N, int M, int D) {
    using namespace attn_bwd;
    const int BH = B * H;
    if (D == 40) return ws_bytes<40, Geo<40>::KT, Geo<40>::WQ, Geo<40>::WK>(BH, N, M);
    if (D == 64) return ws_bytes<64, Geo<64>::KT, Geo<64>::WQ, Geo<64>::WK>(BH, N, M);
    if (D == 80) return ws_bytes<80, Geo<80>::KT, Geo<80>::WQ, Geo<80>::WK>(BH, N, M);
    return 0;
}

// Training backward: dQ, dK, dV from Q, K, V, dO, the forward's LSE and
// Dvec = rowsum(dO * O), all contiguous fp32; ws of at least
// gfx803_fused_attention_train_ws_bytes bytes.
int gfx803_fused_attention_train_bwd(const float* Q, const float* K, const float* V, const float* dO,
                                     const float* LSE, const float* Dvec, float* dQ, float* dK, float* dV,
                                     int B, int H, int N, int M, int D, float scale,
                                     void* ws, size_t ws_size, hipStream_t stream) {
    using namespace attn_bwd;
    if (!Q || !K || !V || !dO || !LSE || !Dvec || !dQ || !dK || !dV || !ws || B <= 0 || H <= 0 || N <= 0 || M <= 0)
        return -1;
    if (D != 40 && D != 64 && D != 80) return -4;
    if (ws_size < gfx803_fused_attention_train_ws_bytes(B, H, N, M, D)) return -5;
    const int BH = B * H;
    float* w = static_cast<float*>(ws);
    (void)hipGetLastError();
    int rc;
    if (D == 40) rc = run<40, Geo<40>::KT, Geo<40>::WQ, Geo<40>::WK>(Q, K, V, dO, LSE, Dvec, dQ, dK, dV, BH, N, M, scale, w, stream);
    else if (D == 64) rc = run<64, Geo<64>::KT, Geo<64>::WQ, Geo<64>::WK>(Q, K, V, dO, LSE, Dvec, dQ, dK, dV, BH, N, M, scale, w, stream);
    else rc = run<80, Geo<80>::KT, Geo<80>::WQ, Geo<80>::WK>(Q, K, V, dO, LSE, Dvec, dQ, dK, dV, BH, N, M, scale, w, stream);
    if (rc != 0) return rc;
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
