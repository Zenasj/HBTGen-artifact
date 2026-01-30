import torch

@torch.compile(dynamic=True)
def f(x, y):
    return y + torch.empty(x).sum()

f(3, torch.randn(3))