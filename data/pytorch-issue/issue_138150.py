import torch

@torch.compile(fullgraph=True)
def foo(x, y, z):
    def f():
        return y + 2
    def g():
        return z + 1
    return torch.cond(x, f, g)

foo(torch.randn(1), torch.randn(3), torch.randn(3))