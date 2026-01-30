import torch

@torch.jit.script
def fails(tensor):
    return torch.isfinite(tensor).all(dim=0)

import torch

@torch.jit.script
def succeeds(tensor):
    return torch.isnan(tensor).all(dim=0)