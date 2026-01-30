import torch

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)