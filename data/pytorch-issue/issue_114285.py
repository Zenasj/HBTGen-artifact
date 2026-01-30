import torch
a = torch.rand((3,3), dtype=torch.float32, device='mps')
torch.linalg.matrix_rank(a)