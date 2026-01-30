import torch
from torch.func import jvp

def func(x):
    y = x.new_tensor(0.5)      # This fails
    # y = x.new_full((), 0.5)  # This works
    return x + y

x = torch.rand(10, 10)
tangents = torch.zeros_like(x)

jvp(func, (x,), (tangents,))