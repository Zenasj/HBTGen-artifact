from torch.fx.experimental.proxy_tensor import make_fx
import torch

def f(x):
    return torch.empty_like(x)

print(make_fx(f)(torch.randn(3)))