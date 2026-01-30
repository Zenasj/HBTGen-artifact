import torch

torch._C._jit_set_profiling_mode(True)
torch._C._jit_set_num_profiled_runs(2)

#...

x1 = torch.randn(4, 8, 8, dtype=torch.float32, device="cuda")
jit_model(x1)
x2 = torch.randn(4, 16, 8, dtype=torch.float32, device="cuda")
jit_model(x2)

#...