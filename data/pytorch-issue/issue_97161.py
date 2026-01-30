import torch

def foo(x):
    y = x.data  # <- Segmentation Fault
    print(y)
    return x

torch.func.vmap(foo)(torch.randn(3, 3))