import torch

input = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 1, dtype=torch.int64, requires_grad=False)
num_groups = 0
weight = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 1, dtype=torch.int64, requires_grad=False)
bias = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 1, dtype=torch.int64, requires_grad=False)
eps = 0
cudnn_enabled = True
torch.group_norm(input, num_groups, weight, bias, eps, cudnn_enabled)