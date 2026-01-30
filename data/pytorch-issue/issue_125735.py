import torch

@torch.compile
def to_float(x):
    return x.to(torch.float32)

x = torch.rand(1024, dtype=torch.half)

y = to_float(x)