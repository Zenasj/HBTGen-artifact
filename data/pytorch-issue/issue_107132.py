import torch

a = torch.randint(1, 500, (4500,), device=torch.device('cuda:0'))
res, index = torch.unique(a, return_inverse=True)