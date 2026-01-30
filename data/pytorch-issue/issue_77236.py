import torch

print("NVFuser enabled ? ", torch._C._jit_nvfuser_enabled())