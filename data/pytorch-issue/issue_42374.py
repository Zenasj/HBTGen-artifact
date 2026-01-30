import torch

a = torch.zeros(1, 2, device='cpu', dtype=torch.int)
b = torch.ones(1, 2, device='cpu', dtype=torch.bfloat16)

res = a.add(b)
print(res)