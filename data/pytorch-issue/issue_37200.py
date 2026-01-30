import torch

@torch.jit.script
def dot(points, query, dim):
    return (points * query).sum(dim)