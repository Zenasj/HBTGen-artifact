import torch

x = torch.zeros(4, device="cuda")
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    x = x + 1

g.reset()
del g

import torch

x = torch.zeros(4, device="cuda")
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    x = x + 1

g.reset()

with torch.cuda.graph(g):
    x = x + 1
g.replay()