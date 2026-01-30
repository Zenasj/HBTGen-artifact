import torch

a = torch.randn(64, dtype=torch.float32)

@torch.compile
def f(a):
    return a.numpy()

f(a)