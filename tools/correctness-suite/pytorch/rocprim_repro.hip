// rocPRIM's select (nonzero's path: lookback scan) and transform (large sort's path) on gfx803,
// outside PyTorch. Prints results against the CPU.
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <cstdio>
#include <vector>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s -> %s\n", #x, hipGetErrorString(e)); return 1; } } while (0)
int main() {
  const unsigned n = 1000;
  std::vector<int> in(n); std::vector<unsigned char> fl(n);
  for (unsigned i = 0; i < n; ++i) { in[i] = i; fl[i] = (i * 7) % 3 == 0; }
  int *din, *dout; unsigned char* dfl; unsigned* dcount; void* tmp = nullptr; size_t tb = 0;
  CK(hipMalloc(&din, n * 4)); CK(hipMalloc(&dout, n * 4)); CK(hipMalloc(&dfl, n)); CK(hipMalloc(&dcount, 4));
  CK(hipMemcpy(din, in.data(), n * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dfl, fl.data(), n, hipMemcpyHostToDevice));
  CK(rocprim::select(tmp, tb, din, dfl, dout, dcount, n));
  CK(hipMalloc(&tmp, tb));
  CK(rocprim::select(tmp, tb, din, dfl, dout, dcount, n));
  CK(hipDeviceSynchronize());
  unsigned c; CK(hipMemcpy(&c, dcount, 4, hipMemcpyDeviceToHost));
  unsigned want = 0; for (auto f : fl) want += f;
  printf("select: count %u (CPU %u)\n", c, want);
  float *fa, *fb; CK(hipMalloc(&fa, n * 4)); CK(hipMalloc(&fb, n * 4));
  CK(rocprim::transform(fa, fb, n, rocprim::identity<float>()));
  CK(hipDeviceSynchronize());
  printf("transform: ok\n");
}
