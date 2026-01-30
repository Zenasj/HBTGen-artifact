import torch

def f(a,b):
    return a+b

opt_f = torch.compile(f)
opt_f(torch.randn(6), torch.randn(6))