import torch

@torch.compile(dynamic=True, fullgraph=True)
def f(rank):
    return torch.ones(10, device=rank.size(0))

f(torch.randn(2))