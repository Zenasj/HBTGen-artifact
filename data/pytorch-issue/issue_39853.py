import torch

@torch.jit.script
def bias_gelu(x):
    return  x * 0.5 * (1.0 + torch.erf(x * 0.70710678))

@torch.jit.script
def bias_gelu_back(g, x):
    ff = 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
    return ff*g