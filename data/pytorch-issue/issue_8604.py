import torch
@torch.jit.script
def fn():
    a = 1
    b = 2
    c = 3
    d = b * c
    return torch.full([1], d)

fn()