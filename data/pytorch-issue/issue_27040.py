import torch

def mul2(x):
     return x*2
torch.jit.trace(mul2, torch.randn(5))