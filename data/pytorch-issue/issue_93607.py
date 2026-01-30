import torch

class A:
    def __setattr__(self):
        pass

@torch.compile(fullgraph=True, backend="eager")
def f(x):
    A()
    return x

f(torch.randn(1))