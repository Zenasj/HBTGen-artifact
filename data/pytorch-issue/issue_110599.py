import torch

py
@torch.compile(backend="eager", fullgraph=True)
def f(x):
    y = set({1, 2, 3})
    if 1 in y:
        return x
    return x - 1

f(x)
x = torch.randn(3)

py
y = set({1, 2, 3})
@torch.compile(backend="eager", fullgraph=True)
def f(x):
     if x in y:
         return 1
     return 0

x = torch.randn(3)
f(x)