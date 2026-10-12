// exclusiveBinaryPrefixScan from PyTorch's ATen/cuda/ScanUtils.cuh (ROCm branch), with its warp
// width as a parameter: 32 is what C10_WARP_SIZE is in gfx8 device code, warpSize is the hardware's.
// Reports how many threads get a wrong index/carry against a CPU prefix sum. No asserts, cannot trap.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <random>

template <int W>
__global__ void scan(const int* in, int* idx, int* carry_out, int n, int use_hw) {
  __shared__ int smem[64];
  const int ww = use_hw ? warpSize : W;
  for (int k = threadIdx.x; k < 64; k += blockDim.x) smem[k] = 0x5a5a;  // stand-in for stale shared memory
  __syncthreads();
  bool v = threadIdx.x < n && in[threadIdx.x];
  unsigned long long vote = __ballot(v);
  unsigned long long le = ~0ull >> (63 - __lane_id());
  int index = __popcll(le & vote), carry = __popcll(vote);
  int warp = threadIdx.x / ww;
  if (__lane_id() == 0) smem[warp] = carry;
  __syncthreads();
  if (threadIdx.x == 0) {
    int cur = 0;
    for (int i = 0; i < blockDim.x / ww; ++i) { int t = smem[i]; smem[i] = t + cur; cur += t; }
  }
  __syncthreads();
  if (warp >= 1) index += smem[warp - 1];
  index -= v;
  idx[threadIdx.x] = index;
  if (threadIdx.x == 0) *carry_out = smem[(blockDim.x + ww - 1) / ww - 1];
}

int main() {
  std::mt19937 rng(1);
  int* d_in; int* d_idx; int* d_c;
  hipMalloc(&d_in, 1024 * 4); hipMalloc(&d_idx, 1024 * 4); hipMalloc(&d_c, 4);
  for (int use_hw = 0; use_hw < 2; ++use_hw) {
    int bad_idx = 0, bad_carry = 0, cases = 0;
    for (int block : {64, 128, 256, 512, 1024}) for (int rep = 0; rep < 20; ++rep) {
      std::vector<int> in(block), idx(block); int c;
      for (auto& x : in) x = rng() % 4 == 0;
      hipMemcpy(d_in, in.data(), block * 4, hipMemcpyHostToDevice);
      hipLaunchKernelGGL(scan<32>, dim3(1), dim3(block), 0, 0, d_in, d_idx, d_c, block, use_hw);
      hipMemcpy(idx.data(), d_idx, block * 4, hipMemcpyDeviceToHost); hipMemcpy(&c, d_c, 4, hipMemcpyDeviceToHost);
      int run = 0; bool bi = false;
      for (int t = 0; t < block; ++t) { if (in[t] && idx[t] != run) bi = true; run += in[t]; }
      bad_idx += bi; bad_carry += c != run; ++cases;
    }
    printf("%s warp width: %d/%d blocks with wrong indices, %d/%d wrong totals\n",
           use_hw ? "hardware (warpSize)" : "C10_WARP_SIZE (32)", bad_idx, cases, bad_carry, cases);
  }
}
