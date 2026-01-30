import torch

@torch.jit.script
def smooth_l1(x, beta: float):
    t = x.abs()
    return torch.where(x < beta, 0.5 * t ** 2 / beta, t - 0.5 * beta)