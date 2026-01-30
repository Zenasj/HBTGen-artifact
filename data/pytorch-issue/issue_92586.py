py
import torch
t = torch.randn(5, requires_grad=True)

def func(t):
    t_copy = t.detach()
    return t_copy

func(t).sum().backward()

py
import torch
t = torch.randn(5, requires_grad=True)

def func(t):
    t_copy = t.detach()
    t_copy.requires_grad = True
    return t_copy

func(t).sum().backward()