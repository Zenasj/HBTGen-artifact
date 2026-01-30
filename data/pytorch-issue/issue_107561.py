import torch

@torch.compile(backend="eager")
def f(x, y):
    return x * 2

f(torch.randn(2), torch.randn(3))