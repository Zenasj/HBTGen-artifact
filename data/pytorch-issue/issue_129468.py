import torch

@torch.compile()
def f(x, y):
    return x @ y

f(torch.rand(2, 2), torch.rand(2, 2))