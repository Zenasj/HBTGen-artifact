import torch

def fn(x):
    return torch.rand(x[-1], len(x))

opt_fn = torch.compile(fn)
opt_fn([4, 5, 6])
opt_fn([7, 8])
opt_fn([9])