import torch

x = torch.randn(10)
z = torch.zeros(10)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    z = x * x
# Warn user