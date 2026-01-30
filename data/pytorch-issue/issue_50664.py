import torch.nn as nn

import torch
import torch.nn.functional as F
import sys

a = torch.randn(1000,1000, requires_grad=True)
b = a
print (f"in: {a.std().item():.4f}")
for i in range(100):
    l = torch.nn.Linear(1000,1000, bias=False)
    torch.nn.init.xavier_normal_(l.weight, torch.nn.init.calculate_gain("selu"))
    b = getattr(F, 'selu')(l(b))
    if i % 10 == 0:
        print (f"out: {b.std().item():.4f}", end=" ")
        a.grad = None
        b.sum().backward(retain_graph=True)
        print (f"grad: {a.grad.abs().mean().item():.4f}")