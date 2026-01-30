import torch

@torch.compile(dynamic=True, backend="eager")
def f(x):
    s0 = x.shape[0]
    y = torch.full((1,), s0)
    return x + y

f(torch.ones(10))
f(torch.ones(11))

import torch

@torch.compile(dynamic=True, backend="inductor")
def f(x):
    s0 = x.shape[0]
    y = torch.full((1,), s0)
    return x + y

f(torch.ones(10))
f(torch.ones(11))