import torch

rand_tensor = torch.rand((2,2)).to('mps')
rand_tensor * 1j