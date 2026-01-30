import torch

@torch.jit.script
def foo(x, y):
    del x, y
    return 3