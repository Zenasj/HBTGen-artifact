import torch

torch.set_default_device('cuda')

@torch.compile
def f(x, y):
    return x.t() + y

f(torch.randn(2**25, 128), torch.randn(128, 2**25))