import torch

def f(a, b):
    return torch.cdist(a, b)

f2 = torch.compile(f)

a = torch.randn(10, 3, requires_grad=True)
b = torch.randn(11, 3, requires_grad=True)
out = f2(a, b)
out.sum().backward()

import torch

def f(a, b):
    return torch.cdist(a, b)

f2 = torch.compile(f, dynamic=True)

a = torch.randn(10, 3, requires_grad=True)
b = torch.randn(11, 3, requires_grad=True)
out = f2(a, b)
out.sum().backward()

aten._cdist_backward