import torch

@torch.compile(backend="aot_eager")
def f(x):
    return torch.mul(x, 1j)


x = torch.randn(4, dtype=torch.complex64, requires_grad=True)
out = f(x)