import torch

a = torch.tensor([[1, 2], [3, 4]])
res = torch.empty_like(a)
torch.gt(a, 2, out=res)