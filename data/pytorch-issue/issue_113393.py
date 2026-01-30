import torch

def fn(x):
    if x.size() != (1, 1, 2, 3):
      return x.cos()
    return x.sin()

torch.compile(fn, backend="eager")(torch.ones(1, 1, 3, 4))
torch.compile(fn, backend="eager")(torch.ones(1, 1, 2, 3))