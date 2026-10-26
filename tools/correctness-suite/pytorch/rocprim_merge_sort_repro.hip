// rocprim::merge_sort (what rocThrust's sort uses for keys that are not plain numbers, such as c10::Half) against a
// host check, with rocPRIM's default and four explicit block configurations. On gfx803 every run that spans more than
// one sorted block (about 1,024 elements) reports keys out of order and values detached from their keys; a correct
// sort reports 0 and 0. hipcc -O2 -std=c++17 --offload-arch=gfx803 rocprim_merge_sort_repro.hip && ./a.out
#include <rocprim/rocprim.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
struct K4 { unsigned v; };
struct Less { __host__ __device__ bool operator()(const K4& a, const K4& b) const { return a.v < b.v; } };
template <class Config> void run(const char* name, int n) {
    std::vector<K4> h(n); std::vector<long> hv(n); srand(1);
    for (int i = 0; i < n; i++) { h[i].v = rand() % 40; hv[i] = i; }
    K4 *din, *dout; long *vin, *vout;
    hipMalloc(&din, n * sizeof(K4)); hipMalloc(&dout, n * sizeof(K4)); hipMalloc(&vin, n * 8); hipMalloc(&vout, n * 8);
    hipMemcpy(din, h.data(), n * sizeof(K4), hipMemcpyHostToDevice); hipMemcpy(vin, hv.data(), n * 8, hipMemcpyHostToDevice);
    size_t bytes = 0; void* tmp = nullptr;
    rocprim::merge_sort<Config>(tmp, bytes, din, dout, vin, vout, (unsigned) n, Less());
    hipMalloc(&tmp, bytes);
    hipError_t e = rocprim::merge_sort<Config>(tmp, bytes, din, dout, vin, vout, (unsigned) n, Less());
    hipDeviceSynchronize();
    std::vector<K4> s(n); std::vector<long> sv(n);
    hipMemcpy(s.data(), dout, n * sizeof(K4), hipMemcpyDeviceToHost); hipMemcpy(sv.data(), vout, n * 8, hipMemcpyDeviceToHost);
    int bad_order = 0, bad_val = 0;
    for (int i = 0; i < n; i++) {
        if (i && s[i].v < s[i - 1].v) bad_order++;
        if (sv[i] < 0 || sv[i] >= n || h[sv[i]].v != s[i].v) bad_val++;
    }
    printf("%-30s n=%6d err=%d  out-of-order %d, values wrong %d\n", name, n, (int) e, bad_order, bad_val);
    hipFree(din); hipFree(dout); hipFree(vin); hipFree(vout); hipFree(tmp);
}
int main() {
    for (int n : {1100, 3000, 300000}) {
        run<rocprim::default_config>("default", n);
        run<rocprim::merge_sort_config<256, 256, 4>>("oddeven 256, sort 256x4", n);
        run<rocprim::merge_sort_config<256, 256, 2>>("oddeven 256, sort 256x2", n);
        run<rocprim::merge_sort_config<512, 512, 1>>("oddeven 512, sort 512x1", n);
        run<rocprim::merge_sort_config<128, 128, 4>>("oddeven 128, sort 128x4", n);
    }
}
