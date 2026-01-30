import torch

def f(x):
    x.detach().mul_(2) # can also happen if the mul_() happens under torch.no_grad()
    return x + 1