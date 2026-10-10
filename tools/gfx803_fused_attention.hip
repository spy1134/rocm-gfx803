#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

#define WARP_SIZE 64

// Fallback naive attention kernel on GPU for arbitrary / unspecialized D (e.g. D=160):
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

// 2D Block-Tiled Fused Attention Kernel for gfx803:
// Tile: Br = 32 queries, Bc = 32 keys/values.
// 64 threads (1 wavefront) per thread block.
// Online softmax tracks running maximum and normalizer in registers.
// Zero register spills to private memory (vgpr_spill_count == 0).
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

    // Cooperatively load Q tile into LDS using coalesced 128-bit float4 loads
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

    // Register accumulators
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
        // Step 1: Load K and V tiles into LDS using coalesced 128-bit float4 loads
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

        // Step 2: Compute 4x4 Score Tile in registers via FMA
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

        // Step 3: Find Row Maximum across 32 columns (reduce across ty)
        float row_m[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            row_m[i] = fmaxf(fmaxf(S[i][0], S[i][1]), fmaxf(S[i][2], S[i][3]));
            #pragma unroll
            for (int mask = 32; mask >= 8; mask >>= 1) {
                row_m[i] = fmaxf(row_m[i], __shfl_xor(row_m[i], mask));
            }
        }

        // Step 4: Online Softmax Update & Exponentials
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

        // Step 5: Store P[4][4] to LDS for V accumulation
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                s_P[row_base + i][col_base + j] = P[i][j];
            }
        }
        __syncthreads();

        // Step 6: Accumulate P * V into acc_O[4][D_PER_TY]
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

    // Step 7: Normalize by l_prev and write out to global DRAM
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
// With Bc=16, LDS footprint is kept <= 32 KiB for D=80 (allowing 2 blocks / 4 waves per CU)
// and <= 43.4 KiB for D=160 (allowing 2 active waves per CU, avoiding the single-wave stall).
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
    constexpr int ROWS_PER_THREAD = (Br / 2) / 8; // (32 / 2) / 8 = 2 rows per thread
    constexpr int COLS_PER_THREAD = Bc / 8;        // 16 / 8 = 2 cols per thread

    const int tid = threadIdx.x;      // 0 .. 127
    const int wave_id = tid >> 6;     // 0 or 1
    const int lane_id = tid & 63;     // 0 .. 63
    const int tx = lane_id & 7;       // 0 .. 7
    const int ty = lane_id >> 3;      // 0 .. 7

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
    if (Q == nullptr || K == nullptr || V == nullptr || O == nullptr) return -1;
    if (B <= 0 || H <= 0 || N <= 0 || M <= 0 || D <= 0) return -2;

    if (D == 40) {
        dim3 block(64);
        dim3 grid((N + 31) / 32, 1, B * H);
        hipLaunchKernelGGL((gfx803_fused_attention_2d_block_kernel<40>), grid, block, 0, stream,
                           Q, K, V, O, B, H, N, M, scale);
    } else if (D == 64) {
        dim3 block(64);
        dim3 grid((N + 31) / 32, 1, B * H);
        hipLaunchKernelGGL((gfx803_fused_attention_2d_block_kernel<64>), grid, block, 0, stream,
                           Q, K, V, O, B, H, N, M, scale);
    } else if (D == 80) {
        dim3 block(128);
        dim3 grid((N + 31) / 32, 1, B * H);
        hipLaunchKernelGGL((gfx803_fused_attention_2wave_kernel<80, 32, 16>), grid, block, 0, stream,
                           Q, K, V, O, B, H, N, M, scale);
    } else if (D == 160) {
        dim3 block(128);
        dim3 grid((N + 31) / 32, 1, B * H);
        hipLaunchKernelGGL((gfx803_fused_attention_2wave_kernel<160, 32, 16>), grid, block, 0, stream,
                           Q, K, V, O, B, H, N, M, scale);
    } else {
        // Fallback for unspecialized head dimensions
        dim3 block(64);
        dim3 grid((N + 63) / 64, 1, B * H);
        hipLaunchKernelGGL(gfx803_fallback_attention_f32, grid, block, 0, stream,
                           Q, K, V, O, B, H, N, M, D, scale);
    }

    return 0;
}

}
