import torch

@torch.jit.script
def f(x):
    return torch.unique(x)

@torch.jit.script
def f(x):
    return torch.clamp(x, min=0)

@torch.jit.script
def f(x):
    return torch.unique(x)

@torch.jit.script
def f(x):
    return torch._unique(x)[0]

@torch.jit.script
def f(x):
    return torch.clamp(x, min=0)

@torch.jit.script
def f(x):
    return torch.clamp_min(x, 0)