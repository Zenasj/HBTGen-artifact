import torch
from torch._higher_order_ops.wrap import wrap

@torch.compile(backend="eager", fullgraph=True)
def f(x):
    return wrap(lambda x: (x.sin(), [x.cos(), x.tan()]), x)

x = torch.tensor(0.3)
output = f(x)