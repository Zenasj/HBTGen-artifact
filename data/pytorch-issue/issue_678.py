import torch
import math

def log_mv_gamma(p, a):
    C = p * (p - 1) / 4 * math.log(math.pi)
    return C + torch.lgamma(a - 0.5 * torch.arange(p, dtype=torch.float)).sum()