# gfx803_accelerate_dtensor_fix.py
# This PyTorch build has no distributed backend compiled in (USE_DISTRIBUTED=0
# -- expected and correct for a single gfx803 card; NCCL/RCCL are irrelevant
# here). accelerate==1.15.0 has a known regression (huggingface/accelerate#4249)
# where Accelerator.prepare_model() unconditionally calls model_has_dtensor(),
# which imports torch.distributed.tensor -- and that import hard-crashes on
# any distributed-disabled PyTorch build, even for plain single-process,
# non-distributed training that could never contain a DTensor in the first
# place. Confirmed as a general accelerate bug, not gfx803/ROCm-specific.
#
# A model on a build with no torch.distributed support cannot contain DTensor
# parameters, so short-circuiting to False is the behaviorally correct answer
# for every case this build can hit -- not just a crash-avoidance hack.
#
# Patched at both the defining module and the accelerator module's own
# namespace, since `from x import y` binds a private reference that patching
# only the source module would miss.
try:
    import accelerate.utils.other as _accel_other

    def _model_has_dtensor_stub(model):
        return False

    _accel_other.model_has_dtensor = _model_has_dtensor_stub

    try:
        import accelerate.accelerator as _accel_accelerator
        if hasattr(_accel_accelerator, "model_has_dtensor"):
            _accel_accelerator.model_has_dtensor = _model_has_dtensor_stub
    except ImportError:
        pass
except Exception:
    # accelerate not installed in this environment -- nothing to patch.
    pass
