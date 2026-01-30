import torch.nn as nn

import torch

dtype = torch.float32

mod_cpu = torch.nn.BatchNorm2d(3, device='cpu')
mod_mps = torch.nn.BatchNorm2d(3, device='mps')

inp_cpu = torch.randn(0, 3, 2, 2, device='cpu', dtype=dtype)
inp_mps = inp_cpu.detach().clone().to('mps')

res_cpu = mod_cpu(inp_cpu)  # passes
res_mps = mod_mps(inp_mps)  # TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.