py
import torch
from torch._higher_order_ops.wrap import wrap

@torch.compile(backend='eager', fullgraph=True, dynamic=True)
def f(x):
    i = x.size(0)
    return wrap(lambda x, i: x.view(i), x, i)

x = torch.randn(3, 1)
f(x)