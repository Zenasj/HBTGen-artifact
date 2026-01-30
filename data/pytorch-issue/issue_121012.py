from contextlib import nullcontext
import torch

@torch.compile(backend='eager', fullgraph=True)
def fn(x, z = None):
    with nullcontext():
        with nullcontext():
            if z is None:
                y = x ** 2
            else:
                y = x ** 3
    return y

x = torch.randn(3)
fn(x)