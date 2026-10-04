# gfx803_torch_distributed_stub_fix.py
# This PyTorch build has no distributed backend compiled in (USE_DISTRIBUTED=0
# -- correct for a single gfx803 card). torch.distributed exists as a bare
# namespace, but several of its attributes (is_initialized,
# destroy_process_group, get_rank, etc.) are never populated, since they're
# normally defined in torch/distributed/distributed_c10d.py, which itself
# fails to import here (same missing torch._C._distributed_c10d extension
# behind the earlier model_has_dtensor crash). accelerate's
# Accelerator.end_training() -> PartialState.destroy_process_group() calls
# torch.distributed.is_initialized() unconditionally during cleanup, even
# for a plain single-process run -- crashing training's teardown after a
# fully successful run.
import torch

_stubs = {
    "is_initialized": lambda: False,
    "is_available": lambda: False,
    "destroy_process_group": lambda *a, **k: None,
    "get_rank": lambda *a, **k: 0,
    "get_world_size": lambda *a, **k: 1,
}
for _name, _fn in _stubs.items():
    if not hasattr(torch.distributed, _name):
        setattr(torch.distributed, _name, _fn)
