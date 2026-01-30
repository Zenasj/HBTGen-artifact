import torch

def make_one_hot(x, i):
    num_events = x.shape[-1]
    return torch.eye(num_events, dtype=x.dtype, device=x.device)[i]