// rocsparse_csrsort (PyTorch reaches it through hipsparseXcsrsort) on CSR matrices whose rows all have one length,
// from 1 to 65,536 entries of distinct random column indices, against std::sort, and the permutation it returns against
// the sorted columns. rocSPARSE's segmented radix sort runs rocPRIM's stable warp sort for rows of up to 4,096 entries;
// built with a compiler that lacks patches/llvm/0004-amdgpu-gfx8-carry-add-sub-clamp.patch it moves indices between
// rows and nearly every such row comes back wrong, rows of one entry included. A correct library prints 0 and 0 on
// every line and the program exits 0.
//   hipcc -O2 --offload-arch=gfx803 rocsparse_csrsort_check.cpp -lrocsparse -o csrsort_check && ./csrsort_check
#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

// A failed set-up call exits with 2, so that it never reads as wrong rows.
#define CHECK_HIP(x) do { if ((x) != hipSuccess) { fprintf(stderr, "%s failed\n", #x); exit(2); } } while (0)
#define CHECK_SPARSE(x) do { if ((x) != rocsparse_status_success) { fprintf(stderr, "%s failed\n", #x); exit(2); } } while (0)

int main() {
    rocsparse_handle h;
    rocsparse_mat_descr d;
    if (rocsparse_create_handle(&h) != rocsparse_status_success || rocsparse_create_mat_descr(&d) != rocsparse_status_success) {
        fprintf(stderr, "rocSPARSE initialisation failed\n");
        return 2;
    }
    std::mt19937 rng(7);
    const int n = 1 << 16;
    std::vector<int> pool(n);
    std::iota(pool.begin(), pool.end(), 0);
    long failures = 0;
    for (int len : {1, 7, 32, 63, 64, 65, 200, 255, 256, 257, 1000, 1024, 1025, 2047, 2048, 2049, 3000, 4095, 4096,
                    4097, 5000, 8191, 8192, 8193, 12000, 16384, 30000, 65536}) {
        const int m = std::max(1, std::min(2000, (1 << 22) / len));
        const int nnz = m * len;
        std::vector<int> ptr(m + 1), col(nnz);
        for (int r = 0; r < m; r++) {
            ptr[r] = r * len;
            std::shuffle(pool.begin(), pool.end(), rng);
            std::copy(pool.begin(), pool.begin() + len, col.begin() + (size_t)r * len);
        }
        ptr[m] = nnz;
        int *dptr, *dcol, *dperm;
        void* buf;
        size_t bs = 0;
        CHECK_HIP(hipMalloc(&dptr, (m + 1) * sizeof(int)));
        CHECK_HIP(hipMalloc(&dcol, (size_t)nnz * sizeof(int)));
        CHECK_HIP(hipMalloc(&dperm, (size_t)nnz * sizeof(int)));
        CHECK_HIP(hipMemcpy(dptr, ptr.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(dcol, col.data(), (size_t)nnz * sizeof(int), hipMemcpyHostToDevice));
        CHECK_SPARSE(rocsparse_csrsort_buffer_size(h, m, n, nnz, dptr, dcol, &bs));
        CHECK_HIP(hipMalloc(&buf, bs ? bs : 4));
        CHECK_SPARSE(rocsparse_create_identity_permutation(h, nnz, dperm));
        const rocsparse_status st = rocsparse_csrsort(h, m, n, nnz, d, dptr, dcol, dperm, buf);
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<int> out(nnz), perm(nnz);
        CHECK_HIP(hipMemcpy(out.data(), dcol, (size_t)nnz * sizeof(int), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(perm.data(), dperm, (size_t)nnz * sizeof(int), hipMemcpyDeviceToHost));
        long bad_rows = 0, bad_perm = 0;
        for (int r = 0; r < m; r++) {
            const size_t b = (size_t)r * len;
            std::vector<int> ref(col.begin() + b, col.begin() + b + len);
            std::sort(ref.begin(), ref.end());
            if (!std::equal(ref.begin(), ref.end(), out.begin() + b)) bad_rows++;
            for (int i = 0; i < len; i++) {
                const int p = perm[b + i];
                if (p < 0 || p >= nnz || col[p] != out[b + i]) {
                    bad_perm++;
                    break;
                }
            }
        }
        printf("row length %6d  rows %5d  status %d  wrong rows %ld  wrong permutation rows %ld\n", len, m, (int)st,
               bad_rows, bad_perm);
        failures += bad_rows + bad_perm + (st != rocsparse_status_success);
        CHECK_HIP(hipFree(dptr));
        CHECK_HIP(hipFree(dcol));
        CHECK_HIP(hipFree(dperm));
        CHECK_HIP(hipFree(buf));
    }
    rocsparse_destroy_mat_descr(d);
    rocsparse_destroy_handle(h);
    return failures ? 1 : 0;
}
