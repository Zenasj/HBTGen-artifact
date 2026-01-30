import torch

@torch.jit.script
def fn(x):
    raise RuntimeError("My error")