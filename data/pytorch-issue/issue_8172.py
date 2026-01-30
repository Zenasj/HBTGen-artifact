import torch

def my_mul(a, b):
    return a * a

inputs = [torch.rand(1), torch.rand(1)]
my_mul_opt = torch.jit.trace(*inputs, optimize=True)(my_mul)

a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

tmp = b * 3

my_mul_opt(a, tmp).sum().backward()