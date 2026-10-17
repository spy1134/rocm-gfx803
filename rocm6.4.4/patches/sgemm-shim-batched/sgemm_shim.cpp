// LD_PRELOAD shim: intercepts rocBLAS's fp32 GEMM entry points and runs the
// standard-algo fp32 path on gfx803_sgemm.h's kernel instead of rocBLAS/Tensile's
// own kernels, which return wrong results on gfx803 (see gfx803_sgemm.h).
//
// Intercepted:
//   rocblas_sgemm
//   rocblas_gemm_ex                     (all-f32 or all-f16, standard algo, C == D)
//   rocblas_sgemm_strided_batched       (torch.bmm / baddbmm -> attention)
//   rocblas_gemm_strided_batched_ex     (all-f32 or all-f16, standard algo, C == D)
// The strided-batched pair matters as much as the plain pair: attention runs
// through it, and its backward pass is wrong on rocBLAS's own kernels.
//
// The two _ex entry points also take fp16 GEMMs (f16 A, B, C and D, fp32
// compute), on the same kernels with fp16 storage, in every layout. rocBLAS
// computes those correctly on gfx803 but slowly: its fp16 kernels for
// op(A) = N with op(B) = T run 15-25x slower than for the other three layouts
// (20.3 ms against 1.1 ms for 1280x4096x320, 510 ms against 23 ms for a
// 16-batch attention GEMM; the layout of every 1x1 convolution's backward
// pass, of MIOpen's other GEMM-based backward convolutions and of attention's
// backward, which doubled an fp16 SD1.5 LoRA training step), and the other
// three layouts run 1.3-1.9x slower than the fp32 kernels here on SD1.5/SDXL's
// linears (2560x8192x320: 3.0 against 4.1 TFLOPS; 1280x512x5120: 1.7 against
// 3.2), because they widen fp16 to fp32 inside the inner loop and load
// operands 2 bytes at a time (see gfx803_sgemm.h). fp16 GEMMs with op(A) = T
// and at most 8 columns of C (nn.Linear in a token-by-token decode loop) run
// on gfx803_sgemm_small_n_kernel: rocBLAS runs them 20-25x slower than
// reading the weight takes.
//
// Everything else (other dtypes, pointer-array batched calls, explicit
// solution_index requests, non-standard algo, device pointer mode, negative
// sizes or leading dimensions too small for the operand) falls through
// untouched to the real rocBLAS symbol via dlsym(RTLD_NEXT, ...), so rocBLAS
// returns its own result or error status for those. m == 0, n == 0 or
// batch_count == 0 return success without launching; alpha == 0 or k == 0
// scale C by beta without reading A or B. Conjugate-transpose is
// treated as transpose, which is identical for real data.
//
// The interception is blanket, not limited to shapes proven bad. Selecting
// Tensile solutions explicitly (rocblas_gemm_algo_solution_index) tests clean
// for min(M,K,N) >= 256, but the default dispatch every real caller uses
// (algo=standard, solution_index=0) still corrupts large shapes such as
// 768x768x3072 by up to 1.4 in magnitude, so explicit-selection results say
// nothing about which shapes are safe.
//
// Runtime switches:
//   GFX803_SGEMM_SHIM_DISABLE=1     -- pass every call straight to rocBLAS
//   GFX803_SGEMM_SHIM_NO_BATCHED=1  -- pass only the strided-batched calls to
//                                      rocBLAS (wrong on gfx803; for bisecting)
//   GFX803_SGEMM_SHIM_NO_FP16=1     -- pass the fp16 calls to rocBLAS
//   GFX803_SGEMM_SHIM_VERBOSE=N     -- log the first N calls (taken over or
//                                      passed through, and why) to stderr
//   GFX803_SGEMM_SMALL_N=N          -- largest n (columns of C) the few-column
//                                      kernel takes, default 8, 0 = off; with 0
//                                      those fp16 calls go to rocBLAS again
//   GFX803_SGEMM_MIN_TILES64=N      -- use the 64x64 (or 32x32) tile from N
//   GFX803_SGEMM_MIN_TILES32=N         tiles of C up (0 = always, -1 = never),
//                                      for benchmarking; see
//                                      gfx803_sgemm_pick_tile for the defaults
//   GFX803_SGEMM_WAVE=N             -- one-wavefront tiles (gfx803_w_kernel)
//                                      for 0: no call, 1: every call given the
//                                      64x64 tile, 3: also those given the
//                                      32x32 tile; unset: gfx803_w_applies's
//                                      rule. Outputs are bit-identical either
//                                      way; for benchmarking
//   GFX803_SGEMM_WAVE_PN=4|8        -- force the 64x32 or 64x64 one-wavefront
//                                      tile instead of the tile-count rule
//   GFX803_SGEMM_WAVE_GW=N          -- tile group width for gfx803_w_kernel
//   GFX803_SGEMM_WAVE_KC2=1         -- fp16 one-wavefront tiles fetch two K
//                                      rows per load
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include "gfx803_sgemm.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

typedef rocblas_status (*rocblas_sgemm_t)(rocblas_handle, rocblas_operation, rocblas_operation,
                                          rocblas_int, rocblas_int, rocblas_int,
                                          const float*, const float*, rocblas_int,
                                          const float*, rocblas_int,
                                          const float*, float*, rocblas_int);

typedef rocblas_status (*rocblas_gemm_ex_t)(rocblas_handle, rocblas_operation, rocblas_operation,
                                            rocblas_int, rocblas_int, rocblas_int,
                                            const void*, const void*, rocblas_datatype, rocblas_int,
                                            const void*, rocblas_datatype, rocblas_int,
                                            const void*, const void*, rocblas_datatype, rocblas_int,
                                            void*, rocblas_datatype, rocblas_int,
                                            rocblas_datatype, rocblas_gemm_algo, int32_t, uint32_t);

typedef rocblas_status (*rocblas_sgemm_strided_batched_t)(
    rocblas_handle, rocblas_operation, rocblas_operation,
    rocblas_int, rocblas_int, rocblas_int,
    const float*,
    const float*, rocblas_int, rocblas_stride,
    const float*, rocblas_int, rocblas_stride,
    const float*,
    float*, rocblas_int, rocblas_stride,
    rocblas_int);

typedef rocblas_status (*rocblas_gemm_strided_batched_ex_t)(
    rocblas_handle, rocblas_operation, rocblas_operation,
    rocblas_int, rocblas_int, rocblas_int,
    const void*,
    const void*, rocblas_datatype, rocblas_int, rocblas_stride,
    const void*, rocblas_datatype, rocblas_int, rocblas_stride,
    const void*,
    const void*, rocblas_datatype, rocblas_int, rocblas_stride,
    void*, rocblas_datatype, rocblas_int, rocblas_stride,
    rocblas_int, rocblas_datatype, rocblas_gemm_algo, int32_t, uint32_t);

static rocblas_sgemm_t real_rocblas_sgemm = nullptr;
static rocblas_gemm_ex_t real_rocblas_gemm_ex = nullptr;
static rocblas_sgemm_strided_batched_t real_rocblas_sgemm_strided_batched = nullptr;
static rocblas_gemm_strided_batched_ex_t real_rocblas_gemm_strided_batched_ex = nullptr;

static void ensure_real_symbols() {
    if (!real_rocblas_sgemm) {
        real_rocblas_sgemm = (rocblas_sgemm_t)dlsym(RTLD_NEXT, "rocblas_sgemm");
    }
    if (!real_rocblas_gemm_ex) {
        real_rocblas_gemm_ex = (rocblas_gemm_ex_t)dlsym(RTLD_NEXT, "rocblas_gemm_ex");
    }
    if (!real_rocblas_sgemm_strided_batched) {
        real_rocblas_sgemm_strided_batched =
            (rocblas_sgemm_strided_batched_t)dlsym(RTLD_NEXT, "rocblas_sgemm_strided_batched");
    }
    if (!real_rocblas_gemm_strided_batched_ex) {
        real_rocblas_gemm_strided_batched_ex =
            (rocblas_gemm_strided_batched_ex_t)dlsym(RTLD_NEXT, "rocblas_gemm_strided_batched_ex");
    }
}

// Escape hatch for A/B testing against the real rocBLAS path without
// rebuilding anything.
static bool env_shim_disabled() {
    const char* v = getenv("GFX803_SGEMM_SHIM_DISABLE");
    return v && v[0] == '1';
}

// Sends only the strided-batched calls back to rocBLAS, so the batched
// kernel can be bisected against the plain GEMM intercepts. Stock rocBLAS
// computes these wrong on gfx803: use it to locate hangs, never to train.
static bool env_batched_disabled() {
    const char* v = getenv("GFX803_SGEMM_SHIM_NO_BATCHED");
    return v && v[0] == '1';
}

static bool env_fp16_disabled() {
    const char* v = getenv("GFX803_SGEMM_SHIM_NO_FP16");
    return v && v[0] == '1';
}

static bool is_f16_call(rocblas_datatype a, rocblas_datatype b, rocblas_datatype c,
                        rocblas_datatype d, rocblas_datatype compute) {
    return a == rocblas_datatype_f16_r && b == rocblas_datatype_f16_r && c == rocblas_datatype_f16_r &&
           d == rocblas_datatype_f16_r && compute == rocblas_datatype_f32_r && !env_fp16_disabled();
}

static int log_limit() {
    static const int limit = [] {
        const char* v = getenv("GFX803_SGEMM_SHIM_VERBOSE");
        return v ? atoi(v) : 0;
    }();
    return limit;
}

static std::atomic<int> g_logged{0};

static bool should_log() {
    int limit = log_limit();
    return limit > 0 && g_logged.fetch_add(1) < limit;
}

static const char* op_name(rocblas_operation op) {
    return op == rocblas_operation_none ? "N" : "T";
}

static void log_call(const char* fn, const char* decision,
                     rocblas_operation tA, rocblas_operation tB,
                     int m, int n, int k, int batch) {
    if (should_log()) {
        fprintf(stderr, "gfx803_sgemm_shim: %s %s%s m=%d n=%d k=%d batch=%d -> %s\n",
                fn, op_name(tA), op_name(tB), m, n, k, batch, decision);
    }
}

// A failed launch is reported, not retried on rocBLAS: the shim launches C in
// column and batch chunks, so earlier chunks may already hold the result, and
// rerunning the whole GEMM would apply them twice when beta != 0, on Tensile
// kernels that compute wrong results on gfx803 to begin with.
static rocblas_status launch_failed(const char* fn, hipError_t err) {
    fprintf(stderr, "gfx803_sgemm_shim: %s kernel launch failed (%s)\n", fn, hipGetErrorString(err));
    return rocblas_status_internal_error;
}

// conjugate-transpose is plain transpose for real data
static bool is_trans(rocblas_operation op) {
    return op != rocblas_operation_none;
}

static bool host_pointer_mode(rocblas_handle handle) {
    rocblas_pointer_mode mode = rocblas_pointer_mode_host;
    if (rocblas_get_pointer_mode(handle, &mode) != rocblas_status_success) return false;
    return mode == rocblas_pointer_mode_host;
}

// Arguments this kernel can compute with; anything else goes to rocBLAS so it
// can report its own error status.
static bool dims_ok(rocblas_operation transA, rocblas_operation transB,
                    int m, int n, int k, int lda, int ldb, int ldc, int batch) {
    if (m < 0 || n < 0 || k < 0 || batch < 0) return false;
    int a_rows = is_trans(transA) ? k : m;
    int b_rows = is_trans(transB) ? n : k;
    auto at_least_one = [](int v) { return v < 1 ? 1 : v; };
    return lda >= at_least_one(a_rows) && ldb >= at_least_one(b_rows) && ldc >= at_least_one(m);
}

extern "C" rocblas_status rocblas_sgemm(rocblas_handle handle,
                                        rocblas_operation transA, rocblas_operation transB,
                                        rocblas_int m, rocblas_int n, rocblas_int k,
                                        const float* alpha,
                                        const float* A, rocblas_int lda,
                                        const float* B, rocblas_int ldb,
                                        const float* beta,
                                        float* C, rocblas_int ldc) {
    ensure_real_symbols();
    const char* why = nullptr;
    if (env_shim_disabled()) why = "passthrough (disabled)";
    else if (!host_pointer_mode(handle)) why = "passthrough (device pointer mode)";
    else if (!dims_ok(transA, transB, m, n, k, lda, ldb, ldc, 1)) why = "passthrough (invalid args)";
    if (why) {
        log_call("sgemm", why, transA, transB, m, n, k, 1);
        return real_rocblas_sgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    if (m == 0 || n == 0) {
        log_call("sgemm", "empty, nothing to do", transA, transB, m, n, k, 1);
        return rocblas_status_success;
    }

    hipStream_t stream = 0;
    rocblas_get_stream(handle, &stream);

    hipError_t herr = gfx803_sgemm((int)m, (int)n, (int)k, *alpha,
                                   A, (int)lda, is_trans(transA),
                                   B, (int)ldb, is_trans(transB),
                                   *beta, C, (int)ldc, stream);
    if (herr != hipSuccess) return launch_failed("sgemm", herr);
    log_call("sgemm", "shim kernel", transA, transB, m, n, k, 1);
    return rocblas_status_success;
}

extern "C" rocblas_status rocblas_gemm_ex(rocblas_handle handle,
                                          rocblas_operation transA, rocblas_operation transB,
                                          rocblas_int m, rocblas_int n, rocblas_int k,
                                          const void* alpha,
                                          const void* a, rocblas_datatype a_type, rocblas_int lda,
                                          const void* b, rocblas_datatype b_type, rocblas_int ldb,
                                          const void* beta,
                                          const void* c, rocblas_datatype c_type, rocblas_int ldc,
                                          void* d, rocblas_datatype d_type, rocblas_int ldd,
                                          rocblas_datatype compute_type, rocblas_gemm_algo algo,
                                          int32_t solution_index, uint32_t flags) {
    ensure_real_symbols();

    bool all_f32 = (a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r &&
                    c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r &&
                    compute_type == rocblas_datatype_f32_r);
    const bool f16 = is_f16_call(a_type, b_type, c_type, d_type, compute_type);
    const char* why = nullptr;
    if (env_shim_disabled()) why = "passthrough (disabled)";
    else if (!all_f32 && !f16) why = "passthrough (not all-f32 or all-f16)";
    else if (algo != rocblas_gemm_algo_standard || solution_index != 0) why = "passthrough (explicit algo/solution)";
    // in-place C==D matches this kernel's beta-accumulate semantics
    else if (c != d || ldc != ldd) why = "passthrough (C != D)";
    else if (!host_pointer_mode(handle)) why = "passthrough (device pointer mode)";
    else if (!dims_ok(transA, transB, m, n, k, lda, ldb, ldd, 1)) why = "passthrough (invalid args)";

    if (why) {
        log_call("gemm_ex", why, transA, transB, m, n, k, 1);
        return real_rocblas_gemm_ex(handle, transA, transB, m, n, k, alpha,
                                    a, a_type, lda, b, b_type, ldb, beta,
                                    c, c_type, ldc, d, d_type, ldd,
                                    compute_type, algo, solution_index, flags);
    }
    if (m == 0 || n == 0) {
        log_call("gemm_ex", "empty, nothing to do", transA, transB, m, n, k, 1);
        return rocblas_status_success;
    }

    hipStream_t stream = 0;
    rocblas_get_stream(handle, &stream);

    // With fp32 compute, alpha and beta are floats for both storage types.
    hipError_t herr = f16
        ? gfx803_gemm_strided_batched<_Float16>((int)m, (int)n, (int)k, *(const float*)alpha,
                                                (const _Float16*)a, (int)lda, 0, is_trans(transA),
                                                (const _Float16*)b, (int)ldb, 0, is_trans(transB),
                                                *(const float*)beta, (_Float16*)d, (int)ldd, 0, 1, stream)
        : gfx803_sgemm((int)m, (int)n, (int)k, *(const float*)alpha,
                       (const float*)a, (int)lda, is_trans(transA),
                       (const float*)b, (int)ldb, is_trans(transB),
                       *(const float*)beta, (float*)d, (int)ldd, stream);
    if (herr != hipSuccess) return launch_failed("gemm_ex", herr);
    log_call("gemm_ex", f16 ? "shim kernel (fp16)" : "shim kernel", transA, transB, m, n, k, 1);
    return rocblas_status_success;
}

extern "C" rocblas_status rocblas_sgemm_strided_batched(rocblas_handle handle,
                                                        rocblas_operation transA,
                                                        rocblas_operation transB,
                                                        rocblas_int m, rocblas_int n, rocblas_int k,
                                                        const float* alpha,
                                                        const float* A, rocblas_int lda,
                                                        rocblas_stride stride_a,
                                                        const float* B, rocblas_int ldb,
                                                        rocblas_stride stride_b,
                                                        const float* beta,
                                                        float* C, rocblas_int ldc,
                                                        rocblas_stride stride_c,
                                                        rocblas_int batch_count) {
    ensure_real_symbols();
    const char* why = nullptr;
    if (env_shim_disabled()) why = "passthrough (disabled)";
    else if (env_batched_disabled()) why = "passthrough (batched disabled)";
    else if (!host_pointer_mode(handle)) why = "passthrough (device pointer mode)";
    else if (!dims_ok(transA, transB, m, n, k, lda, ldb, ldc, batch_count)) why = "passthrough (invalid args)";
    if (why) {
        log_call("sgemm_strided_batched", why, transA, transB, m, n, k, batch_count);
        return real_rocblas_sgemm_strided_batched(handle, transA, transB, m, n, k, alpha,
                                                  A, lda, stride_a, B, ldb, stride_b,
                                                  beta, C, ldc, stride_c, batch_count);
    }
    if (m == 0 || n == 0 || batch_count == 0) {
        log_call("sgemm_strided_batched", "empty, nothing to do", transA, transB, m, n, k, batch_count);
        return rocblas_status_success;
    }

    hipStream_t stream = 0;
    rocblas_get_stream(handle, &stream);

    hipError_t herr = gfx803_sgemm_strided_batched((int)m, (int)n, (int)k, *alpha,
                                                   A, (int)lda, (int64_t)stride_a, is_trans(transA),
                                                   B, (int)ldb, (int64_t)stride_b, is_trans(transB),
                                                   *beta, C, (int)ldc, (int64_t)stride_c,
                                                   (int)batch_count, stream);
    if (herr != hipSuccess) return launch_failed("sgemm_strided_batched", herr);
    log_call("sgemm_strided_batched", "shim kernel", transA, transB, m, n, k, batch_count);
    return rocblas_status_success;
}

extern "C" rocblas_status rocblas_gemm_strided_batched_ex(rocblas_handle handle,
                                                          rocblas_operation transA,
                                                          rocblas_operation transB,
                                                          rocblas_int m, rocblas_int n, rocblas_int k,
                                                          const void* alpha,
                                                          const void* a, rocblas_datatype a_type,
                                                          rocblas_int lda, rocblas_stride stride_a,
                                                          const void* b, rocblas_datatype b_type,
                                                          rocblas_int ldb, rocblas_stride stride_b,
                                                          const void* beta,
                                                          const void* c, rocblas_datatype c_type,
                                                          rocblas_int ldc, rocblas_stride stride_c,
                                                          void* d, rocblas_datatype d_type,
                                                          rocblas_int ldd, rocblas_stride stride_d,
                                                          rocblas_int batch_count,
                                                          rocblas_datatype compute_type,
                                                          rocblas_gemm_algo algo,
                                                          int32_t solution_index, uint32_t flags) {
    ensure_real_symbols();

    bool all_f32 = (a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r &&
                    c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r &&
                    compute_type == rocblas_datatype_f32_r);
    const bool f16 = is_f16_call(a_type, b_type, c_type, d_type, compute_type);
    const char* why = nullptr;
    if (env_shim_disabled()) why = "passthrough (disabled)";
    else if (env_batched_disabled()) why = "passthrough (batched disabled)";
    else if (!all_f32 && !f16) why = "passthrough (not all-f32 or all-f16)";
    else if (algo != rocblas_gemm_algo_standard || solution_index != 0) why = "passthrough (explicit algo/solution)";
    else if (c != d || ldc != ldd || stride_c != stride_d) why = "passthrough (C != D)";
    else if (!host_pointer_mode(handle)) why = "passthrough (device pointer mode)";
    else if (!dims_ok(transA, transB, m, n, k, lda, ldb, ldd, batch_count)) why = "passthrough (invalid args)";

    if (why) {
        log_call("gemm_strided_batched_ex", why, transA, transB, m, n, k, batch_count);
        return real_rocblas_gemm_strided_batched_ex(handle, transA, transB, m, n, k, alpha,
                                                    a, a_type, lda, stride_a,
                                                    b, b_type, ldb, stride_b, beta,
                                                    c, c_type, ldc, stride_c,
                                                    d, d_type, ldd, stride_d,
                                                    batch_count, compute_type, algo,
                                                    solution_index, flags);
    }
    if (m == 0 || n == 0 || batch_count == 0) {
        log_call("gemm_strided_batched_ex", "empty, nothing to do", transA, transB, m, n, k, batch_count);
        return rocblas_status_success;
    }

    hipStream_t stream = 0;
    rocblas_get_stream(handle, &stream);

    hipError_t herr = f16
        ? gfx803_gemm_strided_batched<_Float16>((int)m, (int)n, (int)k, *(const float*)alpha,
                                                (const _Float16*)a, (int)lda, (int64_t)stride_a,
                                                is_trans(transA),
                                                (const _Float16*)b, (int)ldb, (int64_t)stride_b,
                                                is_trans(transB),
                                                *(const float*)beta,
                                                (_Float16*)d, (int)ldd, (int64_t)stride_d,
                                                (int)batch_count, stream)
        : gfx803_sgemm_strided_batched((int)m, (int)n, (int)k, *(const float*)alpha,
                                       (const float*)a, (int)lda, (int64_t)stride_a,
                                       is_trans(transA),
                                       (const float*)b, (int)ldb, (int64_t)stride_b,
                                       is_trans(transB),
                                       *(const float*)beta,
                                       (float*)d, (int)ldd, (int64_t)stride_d,
                                       (int)batch_count, stream);
    if (herr != hipSuccess) return launch_failed("gemm_strided_batched_ex", herr);
    log_call("gemm_strided_batched_ex", f16 ? "shim kernel (fp16)" : "shim kernel", transA, transB, m, n,
             k, batch_count);
    return rocblas_status_success;
}
