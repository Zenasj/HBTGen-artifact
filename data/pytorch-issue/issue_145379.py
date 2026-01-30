import torch

input = torch.tensor(0.3, dtype=torch.float64)
eps = torch.tensor(0.9, dtype=torch.float64)
compiled = torch.compile(torch.logit)
print(f"compiled: {compiled(input, eps)}")
print(f"expected: {torch.log(eps / (1 - eps))}")