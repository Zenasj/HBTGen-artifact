import torch

x[:k] = torch.triangular_solve(b, R[:, :k])
x[k:] = 0