import torch

x = torch.randn(3, 4).cuda()
mask = x.ge(0.5).cuda()

graph = torch.cuda.graphs.CUDAGraph()

with torch.cuda.graph(graph):
    d = x.masked_select(mask)