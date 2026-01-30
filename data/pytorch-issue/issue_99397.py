import torch
x=torch.empty([32, 32]).cuda()
with torch.cuda.graph(torch.cuda.CUDAGraph()):
    torch.matmul(x,x)

import torch

@torch.jit.script
def f(x):
    return 2*x+1

x=torch.randn([32, 32]).cuda()
g0 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g0):
    y = f(x)

g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1):
    y = f(x)

import torch
x=torch.empty([32, 32]).cuda()
torch.matmul(x,x)

with torch.cuda.graph(torch.cuda.CUDAGraph()):
    torch.matmul(x,x)