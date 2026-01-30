import torch

def f(x, n):
    return x + n

x = torch.randn(5, 5)
torch.func.jacrev(f, argnums=0)(x, 1)