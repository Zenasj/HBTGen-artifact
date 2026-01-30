import torch

input = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), -1.5e+300, dtype=torch.float64, requires_grad=False)
weight = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,),
                    1.5e+300, dtype=torch.float64, requires_grad=False)
bias = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 3.5e+35, dtype=torch.float64, requires_grad=False)
N = 1
C = 1
HxW = 1
group = 0
eps = 0
torch.native_group_norm(input, weight, bias, N, C, HxW, group, eps)