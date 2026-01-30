import torch

@torch.compile
def f(*args):
    sym_0, sym_1 = args
    return torch.randint(high=sym_0, size=sym_1)

res = f(0, (3960,))