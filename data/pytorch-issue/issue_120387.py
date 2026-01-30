import torch

@torch.jit.script_if_tracing
def gn(x, y, _ = None):
    return torch.cos(x * y)

@torch.compile(backend="eager")
def fn(x, z):
    return gn(x, 2, z)

print(fn(torch.randn(4), 1))
print(fn(torch.randn(4), torch.randn(4)))