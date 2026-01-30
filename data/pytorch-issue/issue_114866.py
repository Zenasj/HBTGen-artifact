import math
import torch

def func(x, a, b):
    c = 0
    c = c + math.sqrt(a)
    c = c + math.cos(a)
    c = c + math.cosh(a)
    c = c + math.sin(a)
    c = c + math.sinh(a)
    c = c + math.tan(a)
    c = c + math.tanh(a)
    c = c + math.asin(b)
    c = c + math.acos(b)
    c = c + math.atan(a)
    y = x + c
    return y


cfunc = torch.compile(func, dynamic=True, fullgraph=True)

device = "cpu"  # or "cuda"
x = torch.tensor([0, 1, 2, 3], dtype=torch.float32, device=device)
a = 12
b = 1

out = cfunc(x, a, b)
expected = func(x, a, b)
torch.testing.assert_close(out, expected)