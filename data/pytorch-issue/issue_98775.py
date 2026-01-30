import torch
@torch.compile()
def f(): 
    return x + x
x = torch.randn(3)
f()