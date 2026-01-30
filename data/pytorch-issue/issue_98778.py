import torch
x = torch.randn(3)
@torch.compile()
def f():
    return x + x
f()