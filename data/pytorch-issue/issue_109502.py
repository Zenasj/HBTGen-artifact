import torch

@torch.compile(fullgraph=True, backend="eager")
def f(xs):
    if hasattr(xs, 'foo'):
        return xs[0] + 1
    else:
        return xs[0] * 2

f([torch.randn(3)])