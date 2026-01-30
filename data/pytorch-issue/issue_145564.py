import torch

@torch.compile(fullgraph=True, backend="eager")
def f(x):
    x = x.cos()
    def inner():
        return x.sin()
    return inner()

f(torch.ones(10))