import torch
s = torch.sym_int(3)
x = torch.randn((s,))
print(x)

import torch

@torch.compile(backend="aot_eager", dynamic=True)
def f(x):
    return torch.randn(x.shape, generator=None)

x = torch.ones(2)
out = f(x)