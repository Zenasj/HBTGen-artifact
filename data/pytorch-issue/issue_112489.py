import torch

def fn(x, y, z):
    return torch.mm(x, y, out=z)

inputs = [torch.rand((4, 4)) for _ in range(3)]
fn_opt = torch.compile(fn, dynamic=True)
fn_opt(*inputs)