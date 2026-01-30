import torch

@torch.compile
def f(a, b):
    return torch.matmul(a, b)

f(torch.randn(4, 4, 4), torch.randn(1, 4, 4))