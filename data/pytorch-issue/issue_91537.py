import torch

@torch.compile
def f(x):
    return 0.5 * x

f(torch.tensor(1.0))