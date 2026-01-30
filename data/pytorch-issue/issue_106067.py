import torch

@torch.compile(backend="eager")
def f(x, y):
    return y + 2

f(torch.randn(0), torch.randn(2))
f(torch.randn(2), torch.randn(2))