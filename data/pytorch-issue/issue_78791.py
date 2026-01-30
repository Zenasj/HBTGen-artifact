import torch

a = torch.rand(10, 3)
print(torch.argmin(a, dim=None, keepdim=False))  # output: tensor(27)
print(torch.argmin(a, dim=None, keepdim=True))  # output: tensor([[27]])