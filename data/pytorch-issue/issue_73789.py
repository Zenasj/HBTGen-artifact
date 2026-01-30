import torch

@torch.jit.script
def weighted_sum(a, alpha: float, b, beta: float):
    return a * alpha + b * beta