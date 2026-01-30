import torch

py
def myop(x):
    if torch.is_grad_enabled():
        return torch.ops.myop_autograd(x)
    else:
        return torch.ops.myop_noautograd(x)