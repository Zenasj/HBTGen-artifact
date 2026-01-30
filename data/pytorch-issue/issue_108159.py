import torch

@torch.compile(dynamic=True)
def foo(a):
    return a - torch.zeros(3)

foo(4)