import torch

class A:
    pass

@torch.compile(backend='eager', fullgraph=True)
def f(x):
    type(A())
    return x + 1

f(torch.randn(1))