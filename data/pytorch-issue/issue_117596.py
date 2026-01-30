import torch
from torch.testing._internal.two_tensor import TwoTensor, TwoTensorMode

@torch.compile(backend="eager")
def f(x):
    y = x.add(1)
    z = y.a
    return z.mul(3)

x = torch.ones(2)
x_two = TwoTensor(x, x)
out = f(x_two)