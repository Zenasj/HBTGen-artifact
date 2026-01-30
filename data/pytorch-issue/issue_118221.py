py
import torch

def f(x):
    return torch.cond(x > 0, torch.sin, torch.cos, (x,))

x = torch.tensor(1)
f(x)