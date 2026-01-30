import torch

a = 32
b = 256
A = torch.randn(torch.randn(a, b, 1, 1))
A_p = torch.linalg.matrix_power(A, 0)

assert torch.all(A_p == torch.ones(a, b, 1, 1))