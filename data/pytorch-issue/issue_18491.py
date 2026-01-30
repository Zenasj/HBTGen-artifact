import torch

@torch.jit.script
def inner_fn(x):
    return x + x, x + x

def python_op(x):
    return inner_fn(x)

@torch.jit.script
def script_fn(x):
    a, b = python_op(x)
    a.add_(1)
    b.add_(1)
    return a, b

print(script_fn(torch.tensor(1.0)))