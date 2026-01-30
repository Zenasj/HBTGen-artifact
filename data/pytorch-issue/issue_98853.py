import torch

def f(a, b):
    return torch.cdist(a, b)

f2 = torch.compile(f, dynamic=True)

a = torch.randn(10, 3)
b = torch.randn(11, 3)

f(a, b) # works fine
f2(a, b) # errors out