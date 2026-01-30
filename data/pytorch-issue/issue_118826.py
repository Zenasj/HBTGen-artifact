import torch
import torch._higher_order_ops.wrap

glob = []


def f(x):
    glob.append(x)
    return x.clone()

@torch.compile(backend='eager', fullgraph=True)
def g(x):
    return torch.ops.higher_order.wrap(f, x)

x = torch.randn(3)
g(x)