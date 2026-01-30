import math
import torch


def func(x, a):
    b = math.floor(a + 0.5)
    b = math.radians(a) + b
    y = x + b
    return y


cfunc = torch.compile(func, dynamic=True, fullgraph=True, backend="eager")
x = torch.tensor([0, 1, 2, 3], dtype=torch.float32)
a = 12

out = cfunc(x, a)