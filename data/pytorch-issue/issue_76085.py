import torch
from functorch import make_fx
from functorch.experimental import functionalize

def f(x, y):
    return x[y]

t1 = make_fx(functionalize(f))(torch.arange(3), torch.ones(2, dtype=torch.long))
print("Functionalized:\n", t1.graph)