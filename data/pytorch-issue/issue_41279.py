import torch

@torch.jit.script
def dynamic_return_type(cond: bool):
    if cond:
        return 1
    else:
        return "1"