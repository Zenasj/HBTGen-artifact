import torch
from torch.fx.experimental.proxy_tensor import make_fx

def func(a):
    b = a + 1
    c = b.view(-1)
    c.add_(1)
    return b

input = torch.randn(2)
out = make_fx(func)(input)