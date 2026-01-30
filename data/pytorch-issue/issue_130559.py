import torch
import contextlib

@contextlib.contextmanager
def g(x):
    try:
        yield x.sin()
    finally:
        pass

@torch.compile(fullgraph=True)
def fn(x):
    with g(x) as y:
        z = y + 1
    return z

x = torch.randn(2, 3)
fn(x)

# torch._dynamo.exc.Unsupported: 'skip function g in file /micromamba/envs/pytorch/lib/python3.11/contextlib.py'