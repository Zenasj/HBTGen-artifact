import torch

cpu_tensor = torch.mean(torch.rand(2).double()) # float64 tensor on cpu
mps_tensor = torch.mean(torch.rand(2, device='mps')) # float32 tensor on mps

cpu_tensor + mps_tensor # TypeError: unsupported operand type(s) for +: 'Tensor' and 'Tensor'