// BatchNorm training backward (Spatial) correctness sweep. Covers the NCHW
// spatial backward kernel variants that MIOpen picks from H*W and C, with
// and without the saved batch mean / inverse variance (the two paths take
// different code in the kernel and in the solver's key). Shapes straddle
// H*W % 4 (the vectorized-read boundary) and the 1024/4096 variant limits.
// Reference is float64 on the host.
#include "common.hpp"

struct Ref
{
    std::vector<float> dx, dscale, dbias, mean, invvar;
};

static void cpu_bn_bwd(const std::vector<float>& x, const std::vector<float>& dy,
                       const std::vector<float>& scale, int N, int C, int HW, double eps, Ref& r)
{
    r.dx.assign(x.size(), 0.f);
    r.dscale.assign(C, 0.f);
    r.dbias.assign(C, 0.f);
    r.mean.assign(C, 0.f);
    r.invvar.assign(C, 0.f);
    double m = (double)N * HW;
    for(int c = 0; c < C; c++)
    {
        double sum = 0;
        for(int n = 0; n < N; n++)
            for(int i = 0; i < HW; i++)
                sum += x[((size_t)n * C + c) * HW + i];
        double mean = sum / m, var = 0;
        for(int n = 0; n < N; n++)
            for(int i = 0; i < HW; i++)
            {
                double d = x[((size_t)n * C + c) * HW + i] - mean;
                var += d * d;
            }
        var /= m;
        double inv = 1.0 / std::sqrt(var + eps), ds = 0, db = 0;
        for(int n = 0; n < N; n++)
            for(int i = 0; i < HW; i++)
            {
                size_t k = ((size_t)n * C + c) * HW + i;
                db += dy[k];
                ds += dy[k] * (x[k] - mean) * inv;
            }
        for(int n = 0; n < N; n++)
            for(int i = 0; i < HW; i++)
            {
                size_t k    = ((size_t)n * C + c) * HW + i;
                double xhat = (x[k] - mean) * inv;
                r.dx[k]     = (float)(scale[c] * inv / m * (m * dy[k] - db - xhat * ds));
            }
        r.dscale[c] = (float)ds;
        r.dbias[c]  = (float)db;
        r.mean[c]   = (float)mean;
        r.invvar[c] = (float)inv;
    }
}

// vectors_match alone passes a few-percent error on a handful of channels
// (the cosine stays above 0.999), so each output also gets a max-error check
// normalized by the reference's peak magnitude.
static bool check(const char* name, const std::vector<float>& ref, const std::vector<float>& gpu,
                  double* worst_rel)
{
    double metric;
    bool absdiff;
    bool m = vectors_match(ref, gpu, &metric, &absdiff);
    double peak = 0, maxerr = 0;
    for(size_t i = 0; i < ref.size(); i++)
    {
        peak   = std::max(peak, (double)std::fabs(ref[i]));
        double e = std::fabs((double)ref[i] - gpu[i]);
        if(!(e == e))
            e = 1e30; // NaN
        maxerr = std::max(maxerr, e);
    }
    double rel = peak > 0 ? maxerr / peak : maxerr;
    *worst_rel = std::max(*worst_rel, rel);
    bool ok = m && rel < 1e-3;
    printf("    %-6s %s=%.5f maxerr/peak=%.2e%s\n", name, absdiff ? "maxabsdiff" : "cos", metric,
           rel, ok ? "" : "  <-- WRONG");
    return ok;
}

static bool run_one(int N, int C, int H, int W, bool saved)
{
    int HW       = H * W;
    size_t elems = (size_t)N * C * HW;
    double eps   = 1e-5;
    std::mt19937 rng(23 + N * 7 + C * 3 + HW);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> h_x(elems), h_dy(elems), h_scale(C), h_dx(elems), h_ds(C), h_db(C);
    for(auto& v : h_x)
        v = dist(rng) * 1.5f + 0.5f;
    for(auto& v : h_dy)
        v = dist(rng);
    for(auto& v : h_scale)
        v = dist(rng) * 0.5f + 1.0f;
    Ref ref;
    cpu_bn_bwd(h_x, h_dy, h_scale, N, C, HW, eps, ref);

    void *d_x, *d_dy, *d_dx, *d_scale, *d_ds, *d_db, *d_mean = nullptr, *d_inv = nullptr;
    CHECK_HIP(hipMalloc(&d_x, elems * 4));
    CHECK_HIP(hipMalloc(&d_dy, elems * 4));
    CHECK_HIP(hipMalloc(&d_dx, elems * 4));
    CHECK_HIP(hipMalloc(&d_scale, C * 4));
    CHECK_HIP(hipMalloc(&d_ds, C * 4));
    CHECK_HIP(hipMalloc(&d_db, C * 4));
    CHECK_HIP(hipMemcpy(d_x, h_x.data(), elems * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_dy, h_dy.data(), elems * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_scale, h_scale.data(), C * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_dx, 0, elems * 4));
    CHECK_HIP(hipMemset(d_ds, 0, C * 4));
    CHECK_HIP(hipMemset(d_db, 0, C * 4));
    if(saved)
    {
        CHECK_HIP(hipMalloc(&d_mean, C * 4));
        CHECK_HIP(hipMalloc(&d_inv, C * 4));
        CHECK_HIP(hipMemcpy(d_mean, ref.mean.data(), C * 4, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_inv, ref.invvar.data(), C * 4, hipMemcpyHostToDevice));
    }

    miopenHandle_t handle;
    miopenCreate(&handle);
    miopenTensorDescriptor_t xDesc, dyDesc, dxDesc, bnDesc;
    miopenCreateTensorDescriptor(&xDesc);
    miopenCreateTensorDescriptor(&dyDesc);
    miopenCreateTensorDescriptor(&dxDesc);
    miopenCreateTensorDescriptor(&bnDesc);
    miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C, H, W);
    miopenSet4dTensorDescriptor(dyDesc, miopenFloat, N, C, H, W);
    miopenSet4dTensorDescriptor(dxDesc, miopenFloat, N, C, H, W);
    miopenDeriveBNTensorDescriptor(bnDesc, xDesc, miopenBNSpatial);

    float one = 1.0f, zero = 0.0f;
    miopenStatus_t es = miopenBatchNormalizationBackward(
        handle, miopenBNSpatial, &one, &zero, &one, &zero, xDesc, d_x, dyDesc, d_dy, dxDesc, d_dx,
        bnDesc, d_scale, d_ds, d_db, eps, d_mean, d_inv);
    CHECK_HIP(hipDeviceSynchronize());

    bool ok = true;
    printf("N=%d C=%d H=%d W=%d (HW%%4=%d) saved=%d\n", N, C, H, W, HW % 4, (int)saved);
    if(es != miopenStatusSuccess)
    {
        printf("    EXEC FAILED status=%d\n", (int)es);
        ok = false;
    }
    else
    {
        CHECK_HIP(hipMemcpy(h_dx.data(), d_dx, elems * 4, hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_ds.data(), d_ds, C * 4, hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_db.data(), d_db, C * 4, hipMemcpyDeviceToHost));
        double worst = 0;
        ok &= check("dx", ref.dx, h_dx, &worst);
        ok &= check("dscale", ref.dscale, h_ds, &worst);
        ok &= check("dbias", ref.dbias, h_db, &worst);
    }

    hipFree(d_x);
    hipFree(d_dy);
    hipFree(d_dx);
    hipFree(d_scale);
    hipFree(d_ds);
    hipFree(d_db);
    if(saved)
    {
        hipFree(d_mean);
        hipFree(d_inv);
    }
    miopenDestroyTensorDescriptor(xDesc);
    miopenDestroyTensorDescriptor(dyDesc);
    miopenDestroyTensorDescriptor(dxDesc);
    miopenDestroyTensorDescriptor(bnDesc);
    miopenDestroy(handle);
    return ok;
}

int main()
{
    struct Case
    {
        int N, C, H, W;
    };
    Case cases[] = {
        // H*W not divisible by 4
        {1, 8, 33, 33},
        {4, 16, 33, 33},
        {16, 8, 33, 33},
        {2, 16, 65, 65},
        {1, 3, 65, 65},
        {4, 16, 7, 7},
        {8, 5, 17, 19},
        {2, 7, 17, 19},
        {1, 1, 1, 1},
        {16, 13, 1, 1},
        {4, 32, 3, 5},
        {3, 9, 3, 5},
        {1, 7, 1, 1025},
        {2, 7, 1, 1023},
        {2, 6, 25, 41},  // HW = 1025
        {2, 6, 31, 33},  // HW = 1023
        {2, 4, 63, 65},  // HW = 4095
        {2, 4, 65, 63},  // HW = 4095, swapped
        {1, 3, 1, 4097},
        // H*W >= 4096 and divisible by 4 with (N*H*W) % 4096 != 0: the vector path plus its remainder
        {1, 16, 80, 80},
        {1, 16, 96, 96},
        {2, 8, 100, 100},
        {3, 4, 72, 72},
        {4, 6, 72, 72},
        {2, 5, 80, 80},
        {1, 3, 16, 320},
        // H*W exactly at, or divisible-by-4 around, the 1024/4096 boundaries
        {1, 8, 32, 32},
        {4, 17, 32, 32},
        {16, 5, 32, 32},
        {2, 16, 16, 64},
        {1, 8, 64, 64},
        {4, 7, 64, 64},
        {2, 3, 1, 4096},
        {2, 3, 4096, 1},
        {2, 5, 31, 32}, // HW = 992
        {2, 5, 32, 33}, // HW = 1056
        // odd channel counts, N spread
        {1, 1, 14, 14},
        {3, 3, 28, 28},
        {5, 11, 9, 9},
        {7, 31, 10, 10},
        {11, 63, 5, 5},
        {16, 3, 56, 56},
        {16, 127, 4, 4},
        {2, 255, 8, 8},
        {1, 129, 2, 2},
    };
    int total = 0, wrong = 0;
    for(auto& c : cases)
        for(int saved = 1; saved >= 0; saved--)
        {
            total++;
            if(!run_one(c.N, c.C, c.H, c.W, saved))
                wrong++;
        }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG\n", total, wrong);
    return wrong > 0 ? 1 : 0;
}
