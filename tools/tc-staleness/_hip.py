"""ctypes access to the HIP runtime and rocBLAS/rocSOLVER for probe_rocblas.py.

Libraries are opened with dlopen handles (RTLD_LOCAL), so a symbol such as rocblas_sgemm resolves to
librocblas.so itself and not to the LD_PRELOAD sgemm shim. That is deliberate: rocSOLVER reaches rocBLAS
through rocblas_internal_* entry points that the shim does not intercept either.
"""
import ctypes
import os
import sys

import numpy as np

ROCM = os.environ.get("ROCM_LIB", "/opt/rocm/lib")
hip = ctypes.CDLL(f"{ROCM}/libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
rb = ctypes.CDLL(f"{ROCM}/librocblas.so", mode=ctypes.RTLD_GLOBAL)

c_vp = ctypes.c_void_p
c_i = ctypes.c_int
c_f = ctypes.c_float
c_l = ctypes.c_longlong
P = ctypes.POINTER

OP_N, OP_T = 111, 112
UPPER, LOWER = 121, 122
NONUNIT, UNIT = 131, 132
LEFT, RIGHT = 141, 142
EVECT_ORIGINAL, EVECT_NONE = 211, 213

hip.hipMalloc.argtypes = [P(c_vp), ctypes.c_size_t]
hip.hipMemcpy.argtypes = [c_vp, c_vp, ctypes.c_size_t, c_i]
hip.hipMemset.argtypes = [c_vp, c_i, ctypes.c_size_t]
hip.hipFree.argtypes = [c_vp]


def check_hip(rc, what):
    if rc != 0:
        sys.exit(f"STOP {what}: hip error {rc}")


def sync(what):
    check_hip(hip.hipDeviceSynchronize(), "hipDeviceSynchronize after " + what)
    check_hip(hip.hipGetLastError(), "hipGetLastError after " + what)


class Dev:
    """Device buffer that mirrors a numpy array."""

    def __init__(self, arr):
        self.arr = np.asfortranarray(arr)
        self.nbytes = max(self.arr.nbytes, 4)
        self.p = c_vp()
        check_hip(hip.hipMalloc(ctypes.byref(self.p), self.nbytes), "hipMalloc")
        self.put(self.arr)

    def put(self, arr):
        a = np.asfortranarray(arr)
        check_hip(hip.hipMemcpy(self.p, a.ctypes.data_as(c_vp), a.nbytes, 1), "H2D")

    def get(self):
        out = np.empty_like(self.arr, order="F")
        check_hip(hip.hipMemcpy(out.ctypes.data_as(c_vp), self.p, out.nbytes, 2), "D2H")
        return out

    def free(self):
        hip.hipFree(self.p)


def fdev(shape, fill=0.0):
    return Dev(np.full(shape, fill, dtype=np.float32, order="F"))


def idev(shape):
    return Dev(np.zeros(shape, dtype=np.int32, order="F"))


def make_handle():
    h = c_vp()
    rc = rb.rocblas_create_handle(ctypes.byref(h))
    if rc != 0:
        sys.exit(f"STOP rocblas_create_handle rc={rc}")
    return h


def st(rc, what):
    if rc != 0:
        print(f"STATUS {what}: rocblas_status={rc}")
        sys.exit(2)


def header(name):
    print(f"== {name}")
    for k in ("LD_PRELOAD", "GFX803_SGEMM_SHIM_DISABLE", "AMD_SERIALIZE_KERNEL", "HIP_LAUNCH_BLOCKING",
              "HSA_ENABLE_SDMA"):
        print(f"   {k}={os.environ.get(k, '')}")
    sys.stdout.flush()


def rng(seed):
    return np.random.Generator(np.random.PCG64(seed))


def relerr(got, ref):
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    return float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))


def report(op, shape, err, tol, extra=""):
    ok = bool(err <= tol)
    print(f"{'ok   ' if ok else 'WRONG'} {op:<28} {shape:<26} err={err:.3e} tol={tol:.0e} {extra}")
    sys.stdout.flush()
    if not ok:
        print("STOP at first wrong case")
        sys.exit(1)
