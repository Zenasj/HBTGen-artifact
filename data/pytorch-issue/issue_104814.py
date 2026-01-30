import torch

@torch.compile(dynamic=True)
def fn(x, sections):
    return torch.dsplit(x, sections)


fn(torch.randn(4, 4, 4), [1,2,3])