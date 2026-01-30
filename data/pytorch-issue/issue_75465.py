import torch

@torch.jit.script
def foo():
    return torch.arange(-1, 2) == True