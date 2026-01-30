import torch

@torch.jit.script
def foo(a):
    mask = torch.rand(1).byte()
    a.masked_fill(mask, 1)
    return a