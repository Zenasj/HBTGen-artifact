import torch

@torch.compile
def foo(X, Y):
    Z = X + Y
    return Z

X = torch.zeros(10, dtype=torch.complex128)
Y = torch.zeros(10, dtype=torch.complex128)
foo(X, Y)