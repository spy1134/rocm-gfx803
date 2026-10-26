"""opinfo_group.py NAME...: run TestCommonCUDA tests of test_ops.py in this process, one RESULT line per test (see
opinfo_compare_cpu.sh)."""
import sys, os, unittest, time
sys.path.insert(0, "/w/src")
import torch
torch.cuda.set_per_process_memory_fraction(0.6)
import test_ops
class R(unittest.TextTestResult):
    def _emit(self, test, status, detail=""):
        print(f"RESULT {test._testMethodName} {status} {time.time() - self._t0:.1f}s {detail}", flush=True)
    def startTest(self, test):
        self._t0 = time.time(); super().startTest(test)
    def addSuccess(self, test): super().addSuccess(test); self._emit(test, "ok")
    def addFailure(self, test, err):
        super().addFailure(test, err); self._emit(test, "FAIL", str(err[1]).replace("\n", " | ")[:600])
    def addError(self, test, err):
        super().addError(test, err); self._emit(test, "ERROR", (err[0].__name__ + ": " + str(err[1])).replace("\n", " | ")[:600])
    def addSkip(self, test, reason): super().addSkip(test, reason); self._emit(test, "skip", reason[:200])
    def addExpectedFailure(self, test, err): super().addExpectedFailure(test, err); self._emit(test, "xfail")
    def addUnexpectedSuccess(self, test): super().addUnexpectedSuccess(test); self._emit(test, "xpass")
suite = unittest.TestSuite(test_ops.TestCommonCUDA(n) for n in sys.argv[1:])
unittest.TextTestRunner(resultclass=R, verbosity=0, stream=open(os.devnull, "w")).run(suite)
print("GROUPDONE", flush=True)
