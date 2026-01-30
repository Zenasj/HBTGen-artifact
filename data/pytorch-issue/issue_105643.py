py
import torch

@torch.compile(fullgraph=True)
def foo(x):
    values = ()
    values += (x,)

    return values

foo(torch.tensor(2))

py
import torch

@torch.compile(fullgraph=True)
def foo(x):
    values = []
    values += [x]

    return values

foo(torch.tensor(2))

py
import torch

@torch.compile(fullgraph=True)
def foo(x, y):
    values = (y,)
    values += (x,)

    return values

foo(torch.tensor(2), torch.tensor(3))