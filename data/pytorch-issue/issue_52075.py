import torch

torch._C._jit_set_profiling_mode(False) 
torch._C._jit_set_texpr_fuser_enabled(False)