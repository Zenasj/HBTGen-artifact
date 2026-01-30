import torch

def fn(x):
    return x.to(device="cuda", non_blocking=True)

inp = torch.randn(3, 4)

torch.compile(fn)(inp)