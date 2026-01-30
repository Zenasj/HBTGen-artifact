import torch
def fn(x, y):
    torch.abs(x, out=y)

x = torch.rand((8, 8))
y = torch.empty(0)
compiled_fn = torch.compile(fn)
compiled_fn(x, y)
print(y)

import torch
def fn(x, y):
    torch.abs(x, out=y)

x = torch.rand((8, 8))
y = torch.empty(0)
compiled_fn = torch.compile(fn)
compiled_fn(x, y)
print(y)