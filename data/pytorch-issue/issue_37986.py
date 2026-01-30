import torch

@torch.jit.script
def f(x):
    return torch.unique(x)