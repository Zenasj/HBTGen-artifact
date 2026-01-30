import torch

torch._C._jit_set_profiling_mode(True)
torch._C._jit_set_bailout_depth(100)
torch._C._jit_set_num_profiled_runs(1)

torch._C._jit_set_profiling_executor(False)