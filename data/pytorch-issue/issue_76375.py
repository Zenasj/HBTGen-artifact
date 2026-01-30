import torch

def foo(a):
    b = torch.zeros_like(a, dtype=torch.bool)
    return b

a = torch.ones(2, requires_grad=True)
sfoo = torch.jit.script(foo)
sfoo(a)