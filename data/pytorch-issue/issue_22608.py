import torch

@torch.jit.script
def foo(x):
    return torch.ones_like(torch.randn(x), dtype=torch.double)

foo()