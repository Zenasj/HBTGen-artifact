import torch

@torch.compile(dynamic=True)
def f(x):
    assert x.shape[0] > 2
    assert 1 % (x.shape[0] // 2) != 0
    assert 32 * (x.shape[0] // 2) ** 2 - 16 * (x.shape[0] // 2) != 0
    return x.cos()

f(torch.randn(6))

import torch

@torch.compile(dynamic=True)
def f(x):
    torch._check(x.shape[0] > 2)
    torch._check(1 % (x.shape[0] // 2) != 0)
    torch._check(32 * (x.shape[0] // 2) ** 2 - 16 * (x.shape[0] // 2) != 0)
    return x.cos()

f(torch.randn(6))