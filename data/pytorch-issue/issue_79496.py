3
import torch
cuda = torch.device("cuda")
torch.manual_seed(10)
torch.randn(3)
# always yields tensor([-0.6014, -1.0122, -0.3023])
torch.manual_seed(10)
torch.randn(3, device=cuda)
# always yields tensor([-0.1029,  1.6810, -0.2708], device='cuda:0')