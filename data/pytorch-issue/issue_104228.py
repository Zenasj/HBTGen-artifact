import torch

@torch.compile(fullgraph=True, dynamic=True, backend="eager")
def f(x):
    return x << 0

f(2)