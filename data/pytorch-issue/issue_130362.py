py
import torch

@torch.compile(backend="inductor")
def f(x):
    return torch.vmap(torch.mul, in_dims=(0, None))(x, 5)

x = torch.randn(3)
f(x)