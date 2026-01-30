import torch

@torch.jit.script
def foo():
    x : int = 4
    return x