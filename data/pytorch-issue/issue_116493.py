import torch

py
@torch.compile
def f(x: torch.Tensor):
    return torch.diff(x, dim=-1).sum(-1)