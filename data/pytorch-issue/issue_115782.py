import torch
from torch.testing._internal.two_tensor import TwoTensor

@torch.compile
def f(x):
    return x * x
x = TwoTensor(torch.ones(2), torch.ones(2))
x_view = x.view(x.shape)
out = f(x_view)