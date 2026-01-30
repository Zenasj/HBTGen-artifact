import torch

cached = {}
def g(x):
    return x.sin().sin().sin().sin()

@torch.compile(dynamic=False)
def f(x):
    if x.shape not in cached:
        cached[x.shape] = g(x)
    return x + cached[x.shape]

f(torch.randn(10))
f(torch.randn(10))

torch.Size