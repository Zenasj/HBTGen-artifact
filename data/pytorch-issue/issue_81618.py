from contextlib import contextmanager

import torch
from functorch import make_fx
from functorch.experimental import functionalize

def foo(t, y):
    out_1 = torch.ones(1)
    return torch.add(t, y, out=out_1)
    
g = make_fx(functionalize(foo))(torch.tensor([1]), torch.tensor([1]))