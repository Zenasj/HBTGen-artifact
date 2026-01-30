import torch

@torch.compile(backend='eager', fullgraph=True, dynamic=True)
def f(x, xs):
    if x.size(0) is xs:
        return x + 1
    else:
        return x * 2

f(torch.randn(2), 2)