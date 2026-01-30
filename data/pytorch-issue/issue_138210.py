import torch
from torch.fx.experimental.proxy_tensor import make_fx

def foo(x):
    return x + 1

g = make_fx(foo)(torch.randn(3,))