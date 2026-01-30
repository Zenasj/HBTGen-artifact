import torch

d = {}

def fn(x):
    try:
        y = d[0]
    except KeyError:
        y = 1
    return x + y

opt_fn = torch.compile(fn, backend="eager")
opt_fn(torch.randn(3, 3))