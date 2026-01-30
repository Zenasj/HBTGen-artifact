import torch

@torch.compile(dynamic=True)
def f(x):
    return (x + 1).sum()

f(torch.randn(20000, device='cuda'))