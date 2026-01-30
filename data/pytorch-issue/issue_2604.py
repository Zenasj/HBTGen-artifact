import torch

def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(*shape) * (1-eps)
    return -torch.log(-torch.log(U + eps) + eps)