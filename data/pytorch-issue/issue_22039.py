import torch
def foo(x):
    return torch.empty_like(x, dtype=None)
x = torch.empty(3, 3, dtype=torch.int)
foo(x)