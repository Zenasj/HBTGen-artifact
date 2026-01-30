import torch

def add(a, b):
    return a + b

c = torch.compile(add)
c(torch.randn(5, device='cuda'), torch.randn(5, device='cuda'))