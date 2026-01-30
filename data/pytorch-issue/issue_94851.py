import torch.nn as nn

import torch
from torch.func import grad

def f(x):
    return (0.5 * x**2).sum()


x = torch.randn(10)

f(x) # This works
print(torch.allclose(grad(f)(x), x)) # This works

fjit = torch.compile(grad(f))
fjit(x) # Error, see below

import torch
from torch import nn
from torch.func import grad
from torch._dynamo import allow_in_graph
from functools import wraps

def traceable(f):
    f = allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper

class NetworkF(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.W = nn.Parameter(torch.empty((int(4*dim), dim)))
        nn.init.normal_(self.W)
    
    def forward(self, x):
        hid = torch.einsum("...d,yd->...y", x, self.W)
        return hid.sum()
    
f = NetworkF()
x = torch.randn((13,192))

f(x) # Works
grad(f)(x) # Works
traceable(grad(f))(x) # Works
torch.compile(traceable(grad(f)))(x) # BREAKS, details below