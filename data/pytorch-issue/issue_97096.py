import torch

@torch.compile(dynamic=True, backend='eager')
def f(x):
    return x >> 0

f(torch.tensor(0))