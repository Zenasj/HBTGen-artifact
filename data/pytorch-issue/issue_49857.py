import torch
a = torch.tensor([[1,2],[3,0]]) + 0j
print(torch.det(a))