# Loaded by every Python process (PYTHONPATH includes /opt/rocm/lib): the gfx803
# workarounds for PyTorch, torchvision and accelerate, each explained in its own
# module.
import gfx803_nms_fix
import gfx803_layernorm_fix
import gfx803_torch_distributed_stub_fix
import gfx803_accelerate_dtensor_fix
