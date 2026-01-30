import torch

def fn(x):
    b = True
    return x.square() | b

x = torch.tensor([[1, 2], [3, 4]])
fn(x)
torch.jit.trace(fn, (x,))