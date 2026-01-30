import torch

A = torch.full((2, 2), torch.inf)
torch.linalg.eigvals(A)