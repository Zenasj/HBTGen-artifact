import torch

def f(a, b):
    return a[b]

functionalize(foo)(torch.arange(3), torch.ones(2, dtype=torch.long))

def f(a, b):
    return torch.ops.aten.index(a, b)

functionalize(foo)(torch.arange(3), torch.ones(2, dtype=torch.long))