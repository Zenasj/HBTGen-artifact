import torch
from torch._inductor.utils import fresh_inductor_cache

def fn():
    x = torch.randn(1, 10)
    y = torch.randn(10, 1)
    return torch.mm(x, y).sum()

def fn2():
    x = torch.randn(10, 100)
    y = torch.randn(100, 10)
    return torch.mm(x, y).sum()

with fresh_inductor_cache():
    torch.compile(fn)()

torch.compile(fn2)()

import torch
from torch._inductor.utils import fresh_inductor_cache

def fn():
    x = torch.randn(1, 10)
    y = torch.randn(10, 1)
    return torch.mm(x, y).sum()

def fn2():
    x = torch.randn(10, 100)
    y = torch.randn(100, 10)
    return torch.mm(x, y).sum()

with fresh_inductor_cache():
    torch.compile(fn)()

torch.compile(fn2)()