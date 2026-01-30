import torch

a = torch.eye(3)
b = torch.zeros_like(a, layout=torch.sparse_coo)
print(b)