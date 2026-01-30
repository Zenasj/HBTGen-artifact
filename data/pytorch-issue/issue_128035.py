import numpy as np

import torch
def f(a):
    tmp = a.detach()
    a.mul_(2)
    return a, tmp

inp = torch.ones(3, requires_grad=True)
graph_inp = inp.mul(2)

print(inp)
outs = f(graph_inp)
outs[0].sum().backward()
print(inp.grad) # Returns [[4,4,4],[4,4,4],[4,4,4]]

inp = torch.ones(3, requires_grad=True)
graph_inp = inp.mul(2)
print(inp)
outs = torch.compile(f, backend="aot_eager")(graph_inp)
outs[0].sum().backward(retain_graph=True)
print(inp.grad) # Returns [[0,0,0],[0,0,0], [0,0,0]]