import torch

@torch.compile
def f():
    _generator_type = type((_ for _ in ()))

f()

import torch

@torch.compile
def f(x):
    _generator_type = type((_ for _ in ()))
    return x+1

f(torch.randn(10))