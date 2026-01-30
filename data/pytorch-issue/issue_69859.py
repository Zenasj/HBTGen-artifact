import torch

@torch.jit.script
def foo(x, y):
    return torch.sigmoid(x + y)

inp = torch.rand([4, 4])
foo(inp, inp)
foo(inp, inp)
foo(inp, inp)