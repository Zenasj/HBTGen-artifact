import torch

@torch.compile(backend="eager", dynamic=True, fullgraph=True)
def f(x):
    if x.is_contiguous():
        return x
    else:
        return 0

x = torch.randn(13, 14)
f(x)