import torch

def f(x):
    return x.view(torch.int32) >> 2

torch.compile(f)(torch.ones(16, 16))