import torch

@torch.compile(backend="eager", dynamic=True)
def f(x):
    return x + 1

torch.jit.trace(f, torch.randn(3))