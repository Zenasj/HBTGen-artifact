import torch

@torch.jit.script
def fn(x):
    return x


fn(None)