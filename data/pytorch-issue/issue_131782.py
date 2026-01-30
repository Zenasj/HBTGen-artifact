py
import torch

@torch.compile
def f(x):
    return x.sin()

x = torch.randn(3, device='cuda')

_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(_graph):
    y = f(x)

import torch

@torch.compile
def f(x):
    return x.sin()

x = torch.randn(3, device='cuda')

f(x)

_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(_graph):
    y = f(x)