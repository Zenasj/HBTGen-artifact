import torch

@torch.compile
def f(x):
    y = x.to(int)
    return y
x = torch.rand((5,))
print(f(x))