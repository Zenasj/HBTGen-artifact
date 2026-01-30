import torch

def forward(self, x):
    ndim = x.ndim
    # or add, mul, div, etc
    x = torch.sub(x, ndim)
    return x