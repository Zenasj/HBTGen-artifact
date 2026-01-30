import torch

@torch.compile(fullgraph=True)
def f(t):
    return t[t > 0.5]


f(torch.rand(3, 16, 16))