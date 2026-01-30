import torch
from torch._higher_order_ops.wrap import wrap

@torch.compile(backend="eager", fullgraph=True)
def f(x, y):
    return wrap(lambda z: z[0].sin() * z[1].cos(), (x, y))

x = torch.tensor(0.3)
y = torch.tensor(0.6)
output = f(x, y)