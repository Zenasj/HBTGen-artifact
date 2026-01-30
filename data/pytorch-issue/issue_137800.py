import torch

x = torch.rand(16, 16, device='mps', dtype=torch.float16)
y = x[:,0:2].view(torch.float32)
y + 2