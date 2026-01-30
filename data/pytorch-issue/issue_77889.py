import torch

input = torch.full((), 65534, dtype=torch.int64, requires_grad=False)
coefficients = torch.full((5, 1, 5, 5,), 1, dtype=torch.float32, requires_grad=False)
torch._compute_linear_combination(input, coefficients)