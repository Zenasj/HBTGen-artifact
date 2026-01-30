import torch

n = 3
X = torch.randn(n, n, device="cuda")
Y = torch.randn(n,1024*1024, device="cuda")
torch.linalg.solve_triangular(X, Y, upper=True)