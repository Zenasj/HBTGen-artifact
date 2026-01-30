import torch

@torch.jit.script
def works(x: torch.Tensor, y: torch.Tensor):
    x = x.reshape(-1, y.shape[1], y.shape[2])
    return x