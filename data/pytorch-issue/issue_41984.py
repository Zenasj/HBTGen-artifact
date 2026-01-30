import torch

t = torch.tensor((1, 2, 3), device='cuda')
torch.tensor(t, dtype=torch.float64)

t = torch.tensor((1, 2, 3), device='cuda')
torch.tensor(t)