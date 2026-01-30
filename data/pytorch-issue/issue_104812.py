import torch

@torch.compile(dynamic=True)
def fn(x, dim):
    return torch.amin(x, dim=dim, keepdim=True)


fn(torch.randn(4, 4, 4), dim=2)