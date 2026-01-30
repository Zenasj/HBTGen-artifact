import torch
size = 2049
batch = 1
A = torch.rand(batch, size, size, device='cuda')
b = torch.rand(batch, size, device='cuda')
torch.linalg.solve(A, b)