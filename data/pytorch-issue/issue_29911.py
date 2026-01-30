import torch

@torch.jit.script
def baz(c, b):
    return c + b

@torch.jit.script
def foo(c, b):
    return baz(c, b)

@torch.jit.script
def bar(c, b):
    return foo(c, b)

bar(torch.rand(10), torch.rand(9))